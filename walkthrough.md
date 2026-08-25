# Voice Enhancement — Corrected Implementation Walkthrough

## 1. Purpose

This repository implements the speech-enhancement brick for the voicebot pipeline using the **causal speech-enhancement Demucs implementation (`facebookresearch/denoiser`)**, rather than the music-source-separation Demucs models.

It supports two primary paths:

```text
Recorded WAV/MP3
    ↓
16 kHz mono
    ↓
DNS/Demucs speech enhancement
    ↓
Enhanced WAV
```

and:

```text
Live / telephony audio
    ↓
8 kHz PCM
    ↓
8k → 16k
    ↓
stateful DemucsStreamer
    ↓
16k → 8k (when required)
    ↓
streaming ASR
```

The repository is an implementation baseline. It does **not** contain the India-call training corpus or a fine-tuned `dns64_indic` checkpoint.

The upstream `denoiser` repository documents `dns48`, `dns64`, `master64`, `DemucsStreamer`, its dataset JSON format, and fine-tuning from pretrained checkpoints. citeturn0search0

---

# 2. Repository Structure

```text
voice-enhancement/
├── README.md
├── walkthrough.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── data_augmentation.yaml
│   ├── train_dns64_indic.yaml
│   └── infer_streaming.yaml
│
├── data/
│   ├── raw/
│   ├── noise/
│   ├── rirs/
│   └── augmented/
│
├── calls/
│   ├── raw/
│   ├── enhanced/
│   └── held_out/
│
├── models/
│   └── checkpoints/
│
├── baselines/
│   └── demucs_two_stems.py
│
├── scripts/
│   └── prepare_and_train.sh
│
├── src/
│   ├── __init__.py
│   ├── audio_io.py
│   ├── model.py
│   │
│   ├── augment/
│   │   ├── __init__.py
│   │   ├── mix_noise.py
│   │   ├── packet_loss.py
│   │   ├── codec_sim.py
│   │   └── build_dataset.py
│   │
│   ├── infer/
│   │   ├── __init__.py
│   │   ├── batch_infer.py
│   │   └── stream_infer.py
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   └── prepare_denoiser_dataset.py
│   │
│   └── eval/
│       ├── __init__.py
│       ├── benchmark_stream.py
│       └── compute_wer_delta.py
│
└── tests/
    ├── test_packet_loss.py
    └── test_fixes.py
```

---

# 3. What Is Implemented

## Inference

- WAV input.
- MP3 input through the audio loader/FFmpeg fallback.
- Mono conversion.
- Resampling to 16 kHz.
- Pretrained `dns48`, `dns64`, and `master64` model loading.
- Optional local checkpoint loading.
- Batch/offline enhancement.
- Stateful streaming enhancement.
- Live microphone → enhancer → speaker test path.
- 8 kHz telephony adapter.
- 16 kHz model interface.
- Optional conversion back to 8 kHz.
- Dry-signal mixing.
- Stream reset and flush.

## Data generation

- Clean/noise file discovery.
- SNR-controlled additive noise.
- Bursty packet-loss simulation.
- GSM full-rate simulation.
- AMR-NB encode/decode simulation.
- JSONL augmentation manifest.
- Metadata for SNR, packet-loss rate, and codec.

## Training handoff

- Conversion from project JSONL manifest to upstream `denoiser` `clean.json` / `noisy.json` format.
- Training configuration.
- Helper shell script showing the upstream fine-tuning flow.

## Evaluation

- Streaming RTF benchmark.
- Macro WER.
- Pooled/corpus WER.
- Relative WER improvement for both metrics.
- Unit tests for packet loss and the corrected training-manifest format.

---

# 4. Correctness Fixes Applied

The ZIP has been regenerated after fixing the implementation issues identified during code review.

## Fix 1 — `benchmark_stream.py`

### Bug

The previous code used:

```python
load_model(a.model, 'cpu')
```

The second positional argument is `checkpoint`, not `device`. That caused the string `cpu` to be interpreted as a checkpoint path.

### Fixed code

```python
load_model(a.model, device='cpu')
```

The benchmark now correctly loads the pretrained model on CPU.

---

# 5. Fix 2 — Live Streaming Chunking

This was the most important live-audio correctness fix.

`DemucsStreamer.feed()` consumes **new audio** and performs its own internal temporal buffering/stride bookkeeping. It should not receive the entire `total_length` window repeatedly.

## Correct pattern

```text
First call:
    total_length samples

Every subsequent call:
    stride samples
```

The corrected worker now does:

```python
primed = False

required = (
    enhancer.input_frame_samples
    if not primed
    else enhancer.stride_samples
)

frame = buffer[:required]
buffer = buffer[required:]
out = enhancer.process(frame)
primed = True
```

This matches the upstream streaming pattern used by the benchmark: prime once with `total_length`, then advance by `stride`.

The previous implementation instead removed only `stride` samples from the local buffer while feeding a complete `total_length` frame again. That effectively re-submitted already processed audio and could produce duplicated/smeared live output.

---

# 6. Additional Live-Output Fix

The live output callback also had a smaller buffering problem: if a Demucs output chunk was larger than the current sounddevice callback size, the remainder of that chunk was discarded.

The corrected implementation maintains:

```python
self.output_buffer
```

and carries unused samples into the next output callback.

The live path is therefore now:

```text
Microphone callback
        ↓
input queue
        ↓
streaming worker
        ↓
DemucsStreamer
        ↓
output queue
        ↓
output buffer
        ↓
sounddevice callback
        ↓
Speaker
```

This prevents discontinuities caused by dropping the remainder of queued enhanced audio.

---

# 7. Fix 3 — AMR-NB Container Handling

AMR-NB cannot be written as a normal WAV codec/container combination.

The old implementation attempted:

```text
AMR-NB codec
    ↓
out.wav
```

which is invalid.

The corrected implementation uses:

```text
input WAV
    ↓
FFmpeg AMR-NB encoder
    ↓
out.amr
    ↓
FFmpeg decoder
    ↓
decoded.wav
    ↓
rest of pipeline
```

The GSM path remains:

```text
input WAV
    ↓
GSM full-rate in WAV
    ↓
out.wav
```

AMR-NB remains an available codec option but is **not enabled by default** in the augmentation YAML. This avoids making FFmpeg/libopencore-AMR availability a requirement for the default dataset-generation path.

Before enabling it for large-scale generation, verify that the installed FFmpeg build contains the required AMR encoder.

---

# 8. Fix 4 — Upstream Training Manifest Format

This was verified against the upstream `facebookresearch/denoiser` repository.

The upstream `denoiser.audio` tool produces metadata equivalent to:

```python
[
    [file_path, length],
    [file_path, length],
    ...
]
```

The upstream `Audioset` loader explicitly documents that `files` should be a list of `(file, length)` pairs and iterates with:

```python
for file, file_length in self.files:
```

Therefore a list of dictionaries such as:

```json
[
  {"path": "x.wav", "length": 12345}
]
```

would not match the expected format.

The corrected project now writes:

```json
[
  ["x.wav", 12345]
]
```

This is a confirmed correction based on the upstream source. citeturn1view0

---

# 9. Fix 5 — WER Reporting

The previous evaluator calculated:

```text
mean(per-utterance WER)
```

This is a **macro-average WER**.

It is valid as a statistic, but it is not the usual corpus-level pooled WER used for comparing ASR systems.

The corrected evaluator now reports both:

```text
Macro WER
= mean(WER of each utterance)
```

and:

```text
Pooled WER
= total edit distance / total reference words
```

The output now contains:

```json
{
  "n": 100,
  "macro_raw_wer": 0.31,
  "macro_enhanced_wer": 0.25,
  "macro_relative_wer_improvement_pct": 19.3,
  "pooled_raw_wer": 0.29,
  "pooled_enhanced_wer": 0.23,
  "pooled_relative_wer_improvement_pct": 20.7
}
```

For production reporting, use **pooled WER** as the primary corpus-level number, while retaining macro WER for diagnostic analysis.

---

# 10. Batch Audio Path

File:

```text
src/infer/batch_infer.py
```

Example:

```bash
python -m src.infer.batch_infer \
  --in calls/raw/call_00123.mp3 \
  --out calls/enhanced/call_00123.wav \
  --model dns64 \
  --device cpu \
  --dry 0.01
```

Flow:

```text
WAV / MP3
   ↓
load_audio()
   ↓
mono
   ↓
16 kHz
   ↓
Demucs speech denoiser
   ↓
dry mix
   ↓
enhanced WAV
```

No manual padding is required for arbitrary-length file input. The model's internal forward path handles the valid-length padding/cropping requirements.

---

# 11. Live Audio Path

File:

```text
src/infer/stream_infer.py
```

The core class is:

```python
LiveEnhancer
```

It owns one stateful `DemucsStreamer`.

One active call should have its own streamer state:

```text
Call A → LiveEnhancer A
Call B → LiveEnhancer B
Call C → LiveEnhancer C
```

Do not share one streamer between unrelated simultaneous calls.

---

# 12. Live Microphone Test

List audio devices:

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

This is a local hardware test. Production telephony should use the `LiveEnhancer` / `Telephony8kAdapter` API instead of this microphone bridge.

---

# 13. Telephony Integration

The intended production insertion point is:

```text
RTP / WebSocket / Telephony
          ↓
Jitter buffer
          ↓
PLC
          ↓
8 kHz PCM
          ↓
Telephony8kAdapter
          ↓
8k → 16k
          ↓
LiveEnhancer
          ↓
DemucsStreamer
          ↓
16k enhanced audio
          ↓
8k if required by ASR
          ↓
Streaming ASR
```

Example:

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

enhanced_8k = adapter.process(frame_8k)
```

The telephony/RTP/WebSocket layer remains outside this repository because it belongs to the existing voicebot infrastructure.

---

# 14. Dataset Augmentation

File:

```text
src/augment/build_dataset.py
```

Command:

```bash
python -m src.augment.build_dataset \
  --config configs/data_augmentation.yaml \
  --out data/augmented/manifest.jsonl \
  --max-examples 10000
```

Current implemented transformations:

```text
clean speech
   ↓
SNR-controlled additive noise
   ↓
bursty packet-loss simulation
   ↓
optional GSM/AMR-NB codec simulation
   ↓
clean/noisy WAV pair
   ↓
JSONL metadata
```

Each manifest record contains:

```json
{
  "clean": "..._clean.wav",
  "noisy": "..._noisy.wav",
  "snr_db": 5.2,
  "packet_loss_rate": 0.08,
  "codec": "gsm_fr"
}
```

---

# 15. SNR Mixing

`src/augment/mix_noise.py` implements SNR-controlled additive mixing.

The scaling is based on:

```text
scale = RMS(clean)
        ------------------------------
        RMS(noise) × 10^(SNR/20)
```

which is algebraically correct for achieving the requested power/amplitude SNR convention used by the implementation.

---

# 16. Packet Loss

`src/augment/packet_loss.py` simulates bursty loss over configurable frames.

Default frame size:

```text
20 ms
```

Configured loss rates include:

```text
0%
3%
8%
15%
```

The intent is to model bursty telephony/network degradation rather than independently deleting arbitrary samples.

---

# 17. Codec Simulation

`src/augment/codec_sim.py` supports:

```text
none
GSM full-rate
AMR-NB
```

AMR-NB now correctly uses a dedicated `.amr` intermediate container and decodes back to WAV.

The final augmentation manifest still contains WAV paths, so downstream dataset processing does not need to understand AMR containers.

---

# 18. Training Dataset Handoff

Run:

```bash
python -m src.train.prepare_denoiser_dataset \
  --manifest data/augmented/manifest.jsonl \
  --out-dir data/denoiser_egs/tr
```

Generated files:

```text
data/denoiser_egs/tr/
├── clean.json
└── noisy.json
```

Format:

```json
[
  ["/absolute/or/relative/path.wav", 123456],
  ["/absolute/or/relative/path.wav", 987654]
]
```

This matches the upstream `denoiser.audio` / `Audioset` convention. citeturn1view0

---

# 19. Fine-Tuning

The project is designed to fine-tune a pretrained `dns64` model rather than train from scratch.

The upstream repository explicitly supports fine-tuning from:

```text
dns48
dns64
master64
```

and provides the training configuration and dataset-generation workflow. citeturn0search0

The local repository prepares the data and provides the handoff; the actual training loop remains in the upstream `denoiser` project.

No `dns64_indic_v1.th` checkpoint is included yet.

---

# 20. Streaming Benchmark

Run:

```bash
python -m src.eval.benchmark_stream \
  --model dns64 \
  --threads 4 \
  --seconds 10
```

The benchmark now correctly calls:

```python
load_model(a.model, device="cpu")
```

and uses the correct streaming feed pattern:

```text
first feed = total_length
subsequent feeds = stride
```

Benchmark on the actual deployment CPU and under concurrent-call load.

The project plan's target is:

```text
RTF < 0.3
algorithmic latency < 40 ms
```

These are acceptance criteria to validate, not guaranteed values.

---

# 21. WER Evaluation

Your production ASR should generate three JSONL files:

```text
raw.jsonl
enhanced.jsonl
references.jsonl
```

Each record:

```json
{"id": "call_001", "text": "customer transcript"}
```

Run:

```bash
python -m src.eval.compute_wer_delta \
  --raw-transcripts reports/raw.jsonl \
  --enhanced-transcripts reports/enhanced.jsonl \
  --references reports/references.jsonl
```

The evaluator now reports both macro and pooled WER.

For the actual go/no-go decision, pooled WER should be the primary corpus-level metric, supplemented by:

- clean-call regression
- digit/OTP accuracy
- amounts
- dates
- account numbers
- named entities

---

# 22. Tests Run After the Fixes

The corrected repository was syntax-checked and the test suite was run.

Result:

```text
3 passed
```

The tests cover:

- packet-loss tensor behavior
- corrected upstream denoiser manifest shape
- pooled WER calculation

The environment did not have the full upstream `denoiser` runtime installed, so a real neural inference/RTF run was not claimed as completed merely from static tests.

---

# 23. What Is Still Pending

These are intentionally not represented as completed work:

## Data

- Indian clean speech corpus.
- Indian market/street/traffic/crowd noise recordings.
- Real clean call recordings where legally available.
- RIR/reverberation data.

## Training

- Generate a large augmented corpus.
- Fine-tune `dns64`.
- Produce `dns64_indic_v1.th`.
- Validate on held-out real Indian calls.

## Production integration

- Existing RTP/WebSocket/telephony transport.
- Production jitter buffer/PLC.
- Per-call lifecycle and streamer allocation.
- ASR integration.
- Feature flag / bypass control.
- Concurrent-call CPU benchmarking.
- Production metrics and logging.

## Evaluation

- Production-ASR raw-vs-enhanced WER.
- Stratified WER by codec/SNR/packet loss.
- Clean-call regression.
- Digit/entity integrity.
- Latency and RTF under realistic concurrency.

---

# 24. Correct End-to-End Development Sequence

```text
1. Install dependencies
        ↓
2. Run batch WAV/MP3 enhancement with pretrained dns64
        ↓
3. Run local live microphone test
        ↓
4. Connect LiveEnhancer to existing telephony audio path
        ↓
5. Run raw-vs-enhanced ASR WER evaluation
        ↓
6. Collect Indian call/noise data
        ↓
7. Generate augmented manifest
        ↓
8. Prepare upstream denoiser JSON files
        ↓
9. Fine-tune dns64
        ↓
10. Benchmark streaming RTF
        ↓
11. Evaluate pooled WER + entity/digit integrity
        ↓
12. Enable behind a feature flag
        ↓
13. A/B test
        ↓
14. Gradual rollout
```

---

# 25. Final Status

The corrected ZIP should now be treated as the implementation baseline.

### Fixed

- [x] Benchmark CPU/device argument bug.
- [x] Live streamer total-length/stride misuse.
- [x] Live output-buffer data loss.
- [x] AMR-NB invalid WAV-container encoding.
- [x] Upstream training-manifest format.
- [x] Added pooled WER alongside macro WER.

### Verified from source / implementation

- [x] Arbitrary-length batch input relies on model internal valid-length handling.
- [x] SNR mixing formula.
- [x] `(1, T)` mono tensor shape passed into `DemucsStreamer.feed()`.
- [x] Upstream training metadata uses `[path, length]` pairs. citeturn1view0

### Not yet trained / production-ready

- [ ] India-call fine-tuned checkpoint.
- [ ] Large India-call augmentation corpus.
- [ ] Production telephony transport integration.
- [ ] Production concurrency benchmark.
- [ ] Production ASR WER validation.

The architecture remains the same; the regenerated implementation now removes the concrete runtime/correctness issues identified during review.
