# Voice Enhancement — India Call Noise Robustness

Production-oriented speech enhancement brick based on the **causal Demucs speech denoiser (`dns64`/`dns48`)**, not the music-separation Demucs models.

The implementation supports:
- recorded `.wav` / `.mp3` input -> enhanced `.wav`
- live microphone -> enhanced speaker output
- a stateful `LiveEnhancer` API for insertion between telephony/RTP ingestion and streaming ASR
- 8 kHz telephony adapter -> 16 kHz model -> 8 kHz output
- SNR-controlled noise mixing + bursty packet-loss simulation + optional GSM/AMR codec simulation
- manifest generation for later fine-tuning
- ASR (text generation): `ARTPARK-IISc/SraVaani-1.0` batch transcription, wired directly into the
  raw-vs-enhanced pipeline (`src/asr/`)
- WER delta calculation from raw/enhanced ASR transcripts (now generated automatically, not by hand)

The source plan specifically recommends the speech `denoiser` sibling repo and its causal streaming models for live audio; music Demucs is only a throwaway baseline. fileciteturn0file0L13-L37

## 1. Structure

```text
voice-enhancement/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data_augmentation.yaml
│   ├── train_dns64_indic.yaml
│   ├── infer_streaming.yaml
│   └── asr_sravaani.yaml
├── data/
│   ├── raw/
│   ├── noise/
│   ├── rirs/
│   └── augmented/
├── calls/
│   ├── raw/
│   ├── enhanced/
│   └── held_out/
├── models/checkpoints/
├── baselines/
│   └── demucs_two_stems.py
├── scripts/
│   └── prepare_and_train.sh
├── src/
│   ├── audio_io.py
│   ├── model.py
│   ├── augment/
│   │   ├── codec_sim.py
│   │   ├── mix_noise.py
│   │   ├── packet_loss.py
│   │   └── build_dataset.py
│   ├── infer/
│   │   ├── batch_infer.py
│   │   └── stream_infer.py
│   ├── train/
│   │   └── prepare_denoiser_dataset.py
│   ├── asr/
│   │   ├── transcribe.py
│   │   └── pipeline.py
│   └── eval/
│       ├── benchmark_stream.py
│       └── compute_wer_delta.py
└── tests/
    ├── test_packet_loss.py
    └── test_asr.py
```

## 2. Install

### Linux/WSL

```bash
cd voice-enhancement
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install denoiser
sudo apt-get install ffmpeg libsndfile1 portaudio19-dev
```

### Windows PowerShell

```powershell
cd voice-enhancement
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install denoiser
```

Install **FFmpeg** separately and put `ffmpeg.exe` on PATH. It is required for MP3 fallback and codec simulation.

> The upstream `denoiser` project is archived and its published package is old, so use a clean environment and expect dependency compatibility issues on very new Python/Torch combinations. The upstream README supports `pip install denoiser` for pretrained inference and documents `dns48`, `dns64`, `master64`, and streaming evaluation. citeturn0search0turn0search2

## 3. Recorded WAV/MP3 inference

```bash
python -m src.infer.batch_infer \
  --in calls/raw/call_00123.mp3 \
  --out calls/enhanced/call_00123.wav \
  --model dns64 \
  --device cpu \
  --dry 0.01
```

For a local fine-tuned checkpoint:

```bash
python -m src.infer.batch_infer \
  --in calls/raw/call_00123.wav \
  --out calls/enhanced/call_00123.wav \
  --checkpoint models/checkpoints/dns64_indic_v1.th
```

Input is converted to mono 16 kHz before inference. The denoiser models are native 16 kHz. fileciteturn0file0L268-L291

## 4. Live microphone test

List devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Run:

```bash
python -m src.infer.stream_infer \
  --model dns64 \
  --input-device "YOUR_MIC" \
  --output-device "YOUR_SPEAKER" \
  --threads 4 \
  --dry 0.01
```

The streaming wrapper uses the upstream `DemucsStreamer`, which consumes new audio and performs its own internal overlap/stride bookkeeping. The local bridge therefore primes the streamer once with `total_length` samples and then feeds only `stride` samples per hop. fileciteturn0file0L298-L353

## 5. Live telephony/RTP integration

Do **not** route RTP packets directly into the microphone CLI. Use the API:

```python
from src.infer.stream_infer import LiveEnhancer, Telephony8kAdapter

enhancer = LiveEnhancer(
    model_name="dns64",
    device="cpu",
    dry=0.01,
    num_frames=1,
    num_threads=4,
)
adapter = Telephony8kAdapter(enhancer)

# For every decoded 8 kHz mono PCM frame from your jitter buffer/PLC:
enhanced_8k = adapter.process(frame_8k)

# Send enhanced_8k to the streaming ASR frontend.
```

Target path:

```text
RTP / telephony 8 kHz
        ↓
jitter buffer + PLC
        ↓
8 kHz → 16 kHz
        ↓
LiveEnhancer / DemucsStreamer
        ↓
16 kHz → 8 kHz (if ASR expects 8 kHz)
        ↓
streaming ASR
```

This matches the proposed insertion point in the implementation plan. fileciteturn0file0L325-L356

Keep a per-call `enhancement_enabled` flag so you can instantly bypass the enhancer for A/B testing or rollback. fileciteturn0file0L359-L360

## 6. Build India-call augmentation dataset

Put clean Indic speech under the configured `data/raw/*` directories and noise under `data/noise/*`.

```bash
python -m src.augment.build_dataset \
  --config configs/data_augmentation.yaml \
  --out data/augmented/manifest.jsonl \
  --max-examples 10000
```

Each JSONL row contains:

```json
{"clean":"..._clean.wav","noisy":"..._noisy.wav","snr_db":5.2,"packet_loss_rate":0.08,"codec":"gsm_fr"}
```

The plan calls for 8 kHz/telephony degradation, codec simulation, bursty packet loss and controlled additive noise, with metadata retained for stratified evaluation. fileciteturn0file0L133-L156

## 7. Fine-tuning

Prepare denoiser-compatible JSON files. The generated files use the upstream format `[[path, length], ...]`, matching `denoiser.audio`/`Audioset` rather than a list of dictionaries:

```bash
python -m src.train.prepare_denoiser_dataset \
  --manifest data/augmented/manifest.jsonl \
  --out-dir data/denoiser_egs/tr
```

Then clone the upstream training repository outside this project:

```bash
git clone https://github.com/facebookresearch/denoiser ../denoiser
cd ../denoiser
pip install -r requirements.txt
```

Configure its dataset YAML to use this project's generated `clean.json` and `noisy.json`, then start from `dns64` rather than training from scratch:

```bash
./train.py continue_pretrained=dns64 \
  demucs.hidden=64 \
  epochs=40 \
  batch_size=16 \
  lr=3e-5 \
  stft_loss=true
```

The upstream project explicitly documents fine-tuning from `dns48`, `dns64`, or `master64`. citeturn0search0

## 8. Streaming benchmark

```bash
python -m src.eval.benchmark_stream --model dns64 --threads 4 --seconds 10
```

Do this on the actual production CPU and under concurrent-call load. The plan's target is RTF < 0.3 and end-to-end algorithmic latency < 40 ms; those are acceptance targets, not guarantees. fileciteturn0file0L49-L56

## 9. ASR (text generation) -- SraVaani

The text-generation stage is [`ARTPARK-IISc/SraVaani-1.0`](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0):
a ~430M-parameter FastConformer model with a hybrid TDT-CTC decoder, fine-tuned for 65 Indian
languages/dialects. It is a separate model from the causal-Demucs denoiser above -- this stage
transcribes the denoiser's output, it does not replace it.

**Read `src/asr/transcribe.py`'s module docstring before using this in production.** Three things
that are easy to miss:

1. **Gated model.** Accept the access conditions at the model page, then set `HF_TOKEN` (or pass
   `--hf-token`). Without it you get a clear `RuntimeError`, not a cryptic 401 mid-run.
2. **`trust_remote_code=True`.** Loading executes code shipped in the model repo. Review it
   before pointing this at production infra.
3. **Offline/batch, not low-latency streaming.** `model.transcribe(...)` processes a whole
   utterance at once. `StreamingSraVaaniASR` in the same file is an explicit, clearly-labeled
   chunk-and-batch approximation for prototyping only -- it cuts words at chunk boundaries and is
   **not** a real streaming decoder. A real one needs NeMo's cache-aware streaming FastConformer
   path, a materially bigger integration than this wrapper.

Transcribe a directory on its own:

```bash
export HF_TOKEN=hf_your_token_here
python -m src.asr.transcribe --audio-dir calls/raw --out reports/raw.jsonl
```

Or run the full pipeline -- enhance, transcribe both raw and enhanced, and compute the WER delta
in one command, replacing the old "run your production ASR yourself" manual step:

```bash
python -m src.asr.pipeline \
  --raw-dir calls/raw \
  --enhanced-dir calls/enhanced \
  --references reports/references.jsonl \
  --denoiser-checkpoint models/checkpoints/dns64_indic_v1.th
```

`references.jsonl` is ground truth you provide (`{"id":"call_001","text":"..."}`), not generated
by this pipeline. Already have enhanced audio and don't want it regenerated? Add `--skip-enhance`.

Output is the same JSON report as before -- WER is the shipping metric, not PESQ/STOI:

```json
{
  "n": 42,
  "macro_raw_wer": 0.31,
  "macro_enhanced_wer": 0.24,
  "macro_relative_wer_improvement_pct": 22.6,
  "pooled_raw_wer": 0.29,
  "pooled_enhanced_wer": 0.22,
  "pooled_relative_wer_improvement_pct": 24.1
}
```

**SraVaani's WER varies enormously by language** -- its model card shows roughly 12% for Hindi
against 50-78% for many regional dialects. Scope your evaluation to the languages you actually
need, and don't trust an aggregate number across all 65. This is on top of -- not instead of --
the plan's clean-audio regression and digit/OTP entity-integrity checks, which this stage does
not run for you.


## 10. Music-Demucs baseline only

Do not use this as the production denoiser:

```bash
pip install demucs
python baselines/demucs_two_stems.py calls/raw/call_00123.wav
```

The implementation plan identifies music Demucs as a throwaway baseline because it was trained for studio music source separation rather than phone speech denoising. fileciteturn0file0L65-L76

## 11. Important deployment notes

- `dns64` is the default production starting point; use `dns48` if CPU concurrency is too expensive.
- `dry=0.01` is the starting safety setting; change it only after WER evaluation.
- One streamer is stateful per concurrent call; do not share one `DemucsStreamer` between calls.
- Do not claim production readiness until WER, clean-call regression, OTP/digit integrity and concurrent RTF are measured.
- The upstream repository is CC-BY-NC 4.0; verify licensing against your intended commercial deployment before shipping. citeturn0search0
