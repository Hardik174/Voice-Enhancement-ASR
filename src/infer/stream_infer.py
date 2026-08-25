from __future__ import annotations

import argparse
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import torch

from src.model import load_model
import torchaudio
import soundfile as sf

from pathlib import Path


class LiveEnhancer:
    """Stateful streaming enhancer. Input/output are 1-D float32 tensors at 16 kHz."""

    def __init__(self, model_name="dns64", checkpoint=None, device="cpu", dry=0.01, num_frames=1, num_threads=4):
        torch.set_num_threads(num_threads)
        self.device = device
        self.model = load_model(model_name, checkpoint, device)
        from denoiser.demucs import DemucsStreamer
        self.dry = dry
        self.num_frames = num_frames
        self.streamer = DemucsStreamer(self.model, dry=dry, num_frames=num_frames)

    @property
    def sample_rate(self):
        return 16_000

    @property
    def input_frame_samples(self):
        return self.streamer.total_length

    @property
    def stride_samples(self):
        return self.streamer.stride

    def process(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.to(self.device).float().reshape(1, -1)
        with torch.inference_mode():
            out = self.streamer.feed(frame)
        return out.reshape(-1).detach().cpu()

    def flush(self) -> torch.Tensor:
        with torch.inference_mode():
            return self.streamer.flush().reshape(-1).detach().cpu()

    def reset(self):
        from denoiser.demucs import DemucsStreamer
        self.streamer = DemucsStreamer(self.model, dry=self.dry, num_frames=self.num_frames)


class Telephony8kAdapter:
    """Adapt 8 kHz telephony frames to the 16 kHz enhancer and back to 8 kHz."""
    def __init__(self, enhancer: LiveEnhancer):
        import torchaudio
        self.enhancer = enhancer
        self._up = torchaudio.transforms.Resample(8000, 16000)
        self._down = torchaudio.transforms.Resample(16000, 8000)

    def process(self, frame_8k: torch.Tensor) -> torch.Tensor:
        frame_16k = self._up(frame_8k.reshape(1, -1)).reshape(-1)
        out_16k = self.enhancer.process(frame_16k)
        return self._down(out_16k.reshape(1, -1)).reshape(-1)


class AudioStreamBridge:
    """Mic -> enhancer -> speaker. For voicebot use, replace output callback with RTP/WS sink."""

    def __init__(self, enhancer: LiveEnhancer, input_device=None, output_device=None, blocksize=256,
                 save_path="calls/enhanced/live_enhanced.wav", transcribe=False, asr_config=None,
                 transcript_path=None):
        self.enhancer = enhancer
        self.input_device = input_device
        self.output_device = output_device
        self.save_path = save_path
        self.blocksize = blocksize
        # Opt-in: transcribing needs HF_TOKEN + a network call to Hugging Face, so it must not
        # fire for anyone running plain live enhancement without ASR set up.
        self.transcribe = transcribe
        self.asr_config = asr_config
        self.transcript_path = transcript_path
        self.in_q = queue.Queue(maxsize=32)
        self.out_q = queue.Queue(maxsize=64)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.buffer = torch.zeros(0)
        self.output_buffer = np.zeros(0, dtype=np.float32)
        self.output_chunks = []

    def _input_cb(self, indata, frames, time, status):
        if status:
            print(status)
        try:
            self.in_q.put_nowait(torch.from_numpy(indata[:, 0].copy()))
        except queue.Full:
            pass

    def _output_cb(self, outdata, frames, time, status):
        if status:
            print(status)
        outdata.fill(0)

        # Never discard a queued enhanced chunk when the audio callback asks
        # for fewer samples than the chunk contains. Keep the remainder for
        # the next callback so the live stream remains contiguous.
        while len(self.output_buffer) < frames:
            try:
                chunk = self.out_q.get_nowait().numpy().astype(np.float32, copy=False)
            except queue.Empty:
                break
            if len(chunk):
                self.output_buffer = np.concatenate([self.output_buffer, chunk])

        n = min(frames, len(self.output_buffer))
        if n:
            outdata[:n, 0] = self.output_buffer[:n]
            self.output_buffer = self.output_buffer[n:]

    def _worker(self):
        # DemucsStreamer.feed() consumes *new* audio and performs its own
        # internal overlap/stride bookkeeping. Prime it once with total_length
        # samples, then feed only stride-sized chunks. Re-feeding a full window
        # on every hop would duplicate already-processed audio.
        primed = False
        while not self.stop_event.is_set():
            try:
                chunk = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.buffer = torch.cat([self.buffer, chunk])

            required = (
                self.enhancer.input_frame_samples
                if not primed
                else self.enhancer.stride_samples
            )
            while self.buffer.numel() >= required:
                frame = self.buffer[:required]
                self.buffer = self.buffer[required:]
                out = self.enhancer.process(frame)
                primed = True
                required = self.enhancer.stride_samples
                if out.numel():
                    try:
                        self.output_chunks.append(out.detach().cpu())
                        self.out_q.put_nowait(out)
                    except queue.Full:
                        pass

    def run(self):
        self.worker.start()
        print("Live enhancement running. Press Ctrl+C to stop.")
        with sd.InputStream(samplerate=16_000, channels=1, dtype="float32", blocksize=self.blocksize,
                            device=self.input_device, callback=self._input_cb), \
             sd.OutputStream(samplerate=16_000, channels=1, dtype="float32", blocksize=self.blocksize,
                             device=self.output_device, callback=self._output_cb):
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\nStopping live enhancement...")
                self.save_enhanced_audio()
                self.stop_event.set()
                self.worker.join(timeout=2)


    def save_enhanced_audio(self):
        if not self.output_chunks:
            print("No enhanced audio captured.")
            return
    
        audio = torch.cat(self.output_chunks, dim=-1)
        audio = audio.numpy()
    
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    
        sf.write(
            str(path),
            audio,
            16_000,
            subtype="PCM_16",
        )
    
        print(f"Enhanced audio saved to: {path}")

        if self.transcribe:
            self._transcribe_saved_file(path)

    def _transcribe_saved_file(self, path: Path):
        """Transcribe ONLY the one file that was just saved -- never a directory. Imported
        lazily so plain live enhancement (transcribe=False, the default) never needs
        transformers/HF_TOKEN installed or set."""
        from src.asr.transcribe import transcribe_single_file

        out_path = self.transcript_path or str(path.with_suffix(".json"))
        print("Transcribing enhanced audio (this loads SraVaani; may download weights on first run)...")
        try:
            result = transcribe_single_file(str(path), out_path, config=self.asr_config)
        except Exception as exc:  # noqa: BLE001 -- surface any ASR failure without killing the saved audio
            print(f"Transcription failed: {exc}")
            return
        print(f"Transcript saved to: {out_path}")
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description="Live microphone speech enhancement.")
    p.add_argument("--model", default="dns64", choices=["dns48", "dns64", "master64"])
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry", type=float, default=0.01)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--save-path", default="calls/enhanced/live_enhanced.wav")
    p.add_argument("--transcribe", action="store_true",
                    help="After the live session ends, transcribe the saved enhanced audio "
                         "(only that one file) with SraVaani. Requires HF_TOKEN.")
    p.add_argument("--transcript-path", default=None,
                    help="Where to save the transcript JSON. Defaults next to --save-path.")
    p.add_argument("--asr-repo-id", default="ARTPARK-IISc/SraVaani-1.0")
    p.add_argument("--asr-device", default=None)
    p.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var if set")
    args = p.parse_args()
    enhancer = LiveEnhancer(args.model, args.checkpoint, args.device, args.dry, args.num_frames, args.threads)

    asr_config = None
    if args.transcribe:
        from src.asr.transcribe import AsrConfig
        asr_config = AsrConfig(repo_id=args.asr_repo_id, device=args.asr_device, hf_token=args.hf_token)

    AudioStreamBridge(
        enhancer, args.input_device, args.output_device, save_path=args.save_path,
        transcribe=args.transcribe, asr_config=asr_config, transcript_path=args.transcript_path,
    ).run()


if __name__ == "__main__":
    main()
