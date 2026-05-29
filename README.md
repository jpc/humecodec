# humecodec

FFmpeg integration for PyTorch with bundled libraries. Load and save audio/video files directly to PyTorch tensors without requiring a system FFmpeg installation.

## Installation

```bash
pip install humecodec
```

The package includes bundled FFmpeg libraries, so no separate FFmpeg installation is required.

## Quick Start

```pycon
>>> import os, tempfile, math
>>> import torch
>>> import humecodec
>>> from humecodec import MediaDecoder, MediaEncoder, CodecConfig
>>> tmpdir = tempfile.mkdtemp()

```

### Save Audio

```pycon
>>> # Create a 1-second stereo 440 Hz sine wave
>>> sr = 44100
>>> t = torch.arange(sr, dtype=torch.float32) / sr
>>> sine = 0.5 * torch.sin(2 * math.pi * 440 * t)
>>> stereo = torch.stack([sine, sine], dim=0)  # (2, 44100) — channels first
>>> stereo.shape
torch.Size([2, 44100])

>>> # Save as WAV, MP3, and FLAC
>>> wav_path = os.path.join(tmpdir, "test.wav")
>>> mp3_path = os.path.join(tmpdir, "test.mp3")
>>> flac_path = os.path.join(tmpdir, "test.flac")
>>> humecodec.save(wav_path, stereo, sr)
>>> humecodec.save(mp3_path, stereo, sr)
>>> humecodec.save(flac_path, stereo, sr)

```

### Save Compressed Audio (Opus / bitrate)

The output codec is chosen from the file extension. Pass `compression=CodecConfig(bit_rate=...)`
to control the target bitrate for lossy codecs like Opus, MP3, or AAC. Opus always encodes at
48 kHz internally, so the input is resampled automatically.

```pycon
>>> from humecodec import CodecConfig
>>> opus_path = os.path.join(tmpdir, "test.opus")
>>> humecodec.save(opus_path, stereo, sr, compression=CodecConfig(bit_rate=192000))
>>> humecodec.info(opus_path).codec
'opus'

```

`compression` also accepts a bare int/float as shorthand for the bitrate
(`compression=192000` is equivalent to `CodecConfig(bit_rate=192000)`).

### Load Audio

```pycon
>>> # Load returns (waveform, sample_rate) with shape (channel, time)
>>> waveform, loaded_sr = humecodec.load(wav_path)
>>> loaded_sr
44100
>>> waveform.shape  # (channels, time) — stereo
torch.Size([2, 44288])

>>> # Resample to 16 kHz
>>> waveform_16k, sr_16k = humecodec.load(wav_path, sample_rate=16000)
>>> sr_16k
16000

>>> # Slice: skip 0.1s, read 0.5s
>>> sliced, _ = humecodec.load(wav_path, frame_offset=4410, num_frames=22050)
>>> sliced.shape
torch.Size([2, 22050])

>>> # Downmix to mono
>>> mono, _ = humecodec.load(wav_path, num_channels=1)
>>> mono.shape[0]
1

```

### Audio Info

```pycon
>>> ai = humecodec.info(wav_path)
>>> ai.sample_rate
44100
>>> ai.num_channels
2
>>> ai
AudioInfo(sample_rate=44100, num_channels=2, ...)

```

### Streaming Decode

```pycon
>>> decoder = MediaDecoder(wav_path)
>>> decoder.add_audio_stream(frames_per_chunk=4096, buffer_chunk_size=3)
>>> chunks = [c for (c,) in decoder.stream() if c is not None]
>>> len(chunks)
11
>>> chunks[0].shape[1]  # stereo
2

```

### Streaming Encode

```pycon
>>> enc_path = os.path.join(tmpdir, "encoded.wav")
>>> enc = MediaEncoder(enc_path)
>>> enc.add_audio_stream(sample_rate=44100, num_channels=1, format="flt")
>>> chunk = torch.zeros(4096, 1)
>>> with enc.open():
...     enc.write_audio_chunk(0, chunk)
...     enc.write_audio_chunk(0, chunk)
>>> ai = humecodec.info(enc_path)
>>> ai.sample_rate
44100
>>> ai.num_channels
1

```

### Packet Index

```pycon
>>> decoder = MediaDecoder(wav_path)
>>> decoder.add_audio_stream(frames_per_chunk=-1)
>>> index = decoder.build_packet_index(resolution=1)
>>> len(index)
11
>>> entry = index[0]
>>> entry.pts_seconds
0.0
>>> entry.pos  # byte offset
78
>>> entry.size
16384
>>> entry.duration_seconds
0.09287981859410431
>>> total = sum(e.duration_seconds for e in index)
>>> total
1.004263038548753

```

### Custom Filters

```pycon
>>> decoder = MediaDecoder(wav_path)
>>> decoder.add_audio_stream(
...     frames_per_chunk=-1,
...     buffer_chunk_size=-1,
...     filter_desc="aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono",
... )
>>> decoder.process_all_packets()
>>> (filtered,) = decoder.pop_chunks()
>>> filtered.shape[1]  # mono
1

```

### Video

Decode and encode video frames (requires video files, not tested here):

```pycon
# Decode video
decoder = MediaDecoder("video.mp4")
decoder.add_video_stream(frames_per_chunk=1, format="rgb24")

for (frame,) in decoder.stream():
    if frame is not None:
        # frame shape: (1, 3, height, width)
        print(f"Frame at {frame.pts:.2f}s")

# Encode video
encoder = MediaEncoder("output.mp4")
encoder.add_video_stream(
    frame_rate=30.0, width=1920, height=1080,
    format="rgb24", encoder="libx264",
    encoder_option={"crf": "23", "preset": "medium"},
)

with encoder.open():
    for frame in frames:
        encoder.write_video_chunk(0, frame)
```

### Query Functions

```pycon
>>> humecodec.list_audio_backends()
['ffmpeg']

>>> versions = humecodec.get_versions()
>>> sorted(versions.keys())
['libavcodec', 'libavdevice', 'libavfilter', 'libavformat', 'libavutil']

>>> decoders = humecodec.get_audio_decoders()
>>> decoders["mp3"]
'MP3 (MPEG audio layer 3)'
>>> encoders = humecodec.get_audio_encoders()
>>> encoders["flac"]
'FLAC (Free Lossless Audio Codec)'

```

## API Reference

### Convenience Functions

| Function | Description |
|----------|-------------|
| `load(uri, ...)` | Load audio file to tensor `(channel, time)` |
| `save(uri, src, sample_rate, ...)` | Save tensor to audio file |
| `info(path)` | Get audio file metadata (`AudioInfo`) |
| `list_audio_backends()` | Returns `["ffmpeg"]` |

### Classes

| Class | Description |
|-------|-------------|
| `AudioInfo` | Metadata: `sample_rate`, `num_channels`, `num_frames`, `codec`, `format`, `encoding`, `bits_per_sample` |
| `MediaDecoder` | Streaming decoder for audio/video |
| `MediaEncoder` | Streaming encoder for audio/video |
| `CodecConfig` | Codec configuration (`bit_rate`, `compression_level`, `qscale`, `gop_size`, `max_b_frames`) |

### Query Functions

| Function | Description |
|----------|-------------|
| `get_audio_decoders()` | Available audio decoders |
| `get_audio_encoders()` | Available audio encoders |
| `get_video_decoders()` | Available video decoders |
| `get_video_encoders()` | Available video encoders |
| `get_demuxers()` | Available input formats |
| `get_muxers()` | Available output formats |
| `get_versions()` | FFmpeg library versions |

## Tensor Formats

### Audio

- `load()` / `save()` default to **channels-first**: `(channel, time)`
- Pass `channels_first=False` for `(time, channel)`
- dtype: `torch.float32`, range `[-1.0, 1.0]`
- `MediaDecoder` / `MediaEncoder` use `(time, channel)` natively

### Video

- **Shape**: `(num_frames, channels, height, width)`
- **dtype**: `torch.uint8` for RGB/BGR, `torch.float32` for YUV
- RGB24: `(N, 3, H, W)`, values `[0, 255]`

## Building from Source

```bash
git clone https://github.com/your-org/humecodec
cd humecodec

# Editable install (auto-fetches prebuilt FFmpeg if needed)
pip install -e .

# Or with custom FFmpeg location
HUMECODEC_FFMPEG_ROOT=/path/to/ffmpeg pip install -e .
```

## License

BSD-3-Clause

This project bundles FFmpeg libraries which are licensed under LGPL/GPL. See FFmpeg's license for details.
