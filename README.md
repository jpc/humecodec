# humecodec

FFmpeg integration for PyTorch with bundled libraries. Load and save audio/video files directly to PyTorch tensors without requiring a system FFmpeg installation.

## Installation

```bash
pip install humecodec
```

The package includes bundled FFmpeg libraries, so no separate FFmpeg installation is required.

## Quick Start

### Load Audio

```python
import humecodec

# Load an audio file (returns tensor and sample rate)
waveform, sample_rate = humecodec.load_audio("audio.mp3")
print(f"Shape: {waveform.shape}")  # (num_frames, num_channels)
print(f"Sample rate: {sample_rate}")

# Load with resampling
waveform, sr = humecodec.load_audio("audio.mp3", sample_rate=16000)

# Load a specific duration starting from an offset
waveform, sr = humecodec.load_audio("audio.mp3", offset=1.0, duration=5.0)

# Load as mono
waveform, sr = humecodec.load_audio("audio.mp3", num_channels=1)
```

### Save Audio

```python
import torch
import humecodec

# Create a simple sine wave
sample_rate = 44100
duration = 2.0
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = 0.5 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(1)  # 440 Hz tone

# Save as WAV
humecodec.save_audio("output.wav", waveform, sample_rate)

# Save as MP3
humecodec.save_audio("output.mp3", waveform, sample_rate)

# Save as FLAC with custom encoder options
humecodec.save_audio(
    "output.flac",
    waveform,
    sample_rate,
    encoder_option={"compression_level": "8"}
)
```

### Get Audio Info

```python
import humecodec

info = humecodec.info("audio.mp3")
print(f"Sample rate: {info.sample_rate}")
print(f"Channels: {info.num_channels}")
print(f"Duration: {info.num_frames / info.sample_rate:.2f}s")
print(f"Codec: {info.codec}")
```

## Advanced Usage

### Streaming Decode

For large files or real-time processing, use the streaming API:

```python
from humecodec import MediaDecoder

decoder = MediaDecoder("long_audio.wav")
decoder.add_audio_stream(
    frames_per_chunk=4096,  # Process 4096 frames at a time
    buffer_chunk_size=3,
)

for (chunk,) in decoder.stream():
    if chunk is not None:
        # Process chunk: shape (frames_per_chunk, num_channels)
        process(chunk)
        print(f"PTS: {chunk.pts:.2f}s")
```

### Streaming Encode

```python
from humecodec import MediaEncoder
import torch

encoder = MediaEncoder("output.wav")
encoder.add_audio_stream(
    sample_rate=44100,
    num_channels=2,
    format="flt",  # 32-bit float input
)

with encoder.open():
    # Write audio in chunks
    for chunk in generate_audio_chunks():
        encoder.write_audio_chunk(0, chunk)
```

### Video Support

```python
from humecodec import MediaDecoder, MediaEncoder

# Decode video
decoder = MediaDecoder("video.mp4")
decoder.add_video_stream(
    frames_per_chunk=1,
    format="rgb24",  # Output as RGB
)

for (frame,) in decoder.stream():
    if frame is not None:
        # frame shape: (1, 3, height, width)
        print(f"Frame at {frame.pts:.2f}s")

# Encode video
encoder = MediaEncoder("output.mp4")
encoder.add_video_stream(
    frame_rate=30.0,
    width=1920,
    height=1080,
    format="rgb24",
    encoder="libx264",
    encoder_option={"crf": "23", "preset": "medium"},
)

with encoder.open():
    for frame in frames:
        # frame shape: (1, 3, height, width), dtype uint8
        encoder.write_video_chunk(0, frame)
```

### Custom Filter Graphs

Apply FFmpeg filters during decode:

```python
from humecodec import MediaDecoder

decoder = MediaDecoder("audio.wav")

# Add audio stream with filter (resample + convert to mono)
decoder.add_audio_stream(
    frames_per_chunk=-1,  # Read all at once
    buffer_chunk_size=-1,
    filter_desc="aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono",
)

decoder.process_all_packets()
chunks = decoder.pop_chunks()
waveform = chunks[0]  # Resampled mono audio
```

### Seeking

```python
from humecodec import MediaDecoder

decoder = MediaDecoder("audio.mp3")
decoder.add_audio_stream(frames_per_chunk=44100)

# Seek to 30 seconds
decoder.seek(30.0, mode="precise")  # or "key" for keyframe-only

for (chunk,) in decoder.stream():
    # Chunks start from ~30s
    print(f"PTS: {chunk.pts}")
```

## API Reference

### Convenience Functions

| Function | Description |
|----------|-------------|
| `load_audio(path, ...)` | Load audio file to tensor |
| `save_audio(path, waveform, sample_rate, ...)` | Save tensor to audio file |
| `info(path)` | Get audio file metadata |

### Classes

| Class | Description |
|-------|-------------|
| `MediaDecoder` | Streaming decoder for audio/video |
| `MediaEncoder` | Streaming encoder for audio/video |
| `CodecConfig` | Codec configuration (bit_rate, gop_size, etc.) |

### Query Functions

```python
import humecodec

# List available codecs
humecodec.get_audio_decoders()  # {'mp3': 'MP3 ...', 'aac': 'AAC ...', ...}
humecodec.get_audio_encoders()
humecodec.get_video_decoders()
humecodec.get_video_encoders()

# List available formats
humecodec.get_demuxers()  # Input formats
humecodec.get_muxers()    # Output formats

# Get FFmpeg library versions
humecodec.get_versions()
# {'libavcodec': (62, 11, 100), 'libavformat': (62, 3, 100), ...}
```

## Tensor Formats

### Audio

- **Shape**: `(num_frames, num_channels)`
- **dtype**: `torch.float32` (default), range `[-1.0, 1.0]`
- Stereo: `(N, 2)`, Mono: `(N, 1)`

### Video

- **Shape**: `(num_frames, channels, height, width)`
- **dtype**: `torch.uint8` for RGB/BGR, `torch.float32` for YUV
- RGB24: `(N, 3, H, W)`, values `[0, 255]`

## Supported Formats

The bundled FFmpeg includes support for common formats:

**Audio**: WAV, MP3, AAC, FLAC, OGG/Vorbis, Opus
**Video**: H.264, H.265/HEVC, VP8, VP9, AV1
**Containers**: MP4, MKV, WebM, AVI, MOV

## Building from Source

For development or custom FFmpeg builds:

```bash
git clone https://github.com/your-org/humecodec
cd humecodec

# Install with system FFmpeg
pip install -e .

# Or with custom FFmpeg location
HUMECODEC_FFMPEG_ROOT=/path/to/ffmpeg pip install -e .
```

### Building Wheels Locally

To build manylinux wheels with bundled FFmpeg libraries using Docker:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheel for current Python version (e.g., cp310)
sudo CIBW_MANYLINUX_X86_64_IMAGE=quay.io/pypa/manylinux_2_28_x86_64 \
    cibuildwheel --only cp310-manylinux_x86_64 --output-dir wheelhouse

# Build all Python versions
sudo CIBW_MANYLINUX_X86_64_IMAGE=quay.io/pypa/manylinux_2_28_x86_64 \
    cibuildwheel --output-dir wheelhouse
```

The resulting wheel (~38 MB) includes all FFmpeg libraries and works without any system FFmpeg installation.

## License

BSD-3-Clause

This project bundles FFmpeg libraries which are licensed under LGPL/GPL. See FFmpeg's license for details.
