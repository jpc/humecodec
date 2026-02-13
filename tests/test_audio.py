"""Round-trip encode/decode tests for humecodec."""

import math
import torch
import pytest
import humecodec


def _make_sine(sample_rate: int = 44100, duration: float = 1.0,
               freq: float = 440.0, num_channels: int = 1) -> torch.Tensor:
    """Generate a sine-wave tensor of shape (num_frames, num_channels)."""
    n = int(sample_rate * duration)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    wave = 0.5 * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(1).expand(-1, num_channels).contiguous()


def test_save_load_wav_roundtrip(tmp_path):
    path = str(tmp_path / "test.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)

    humecodec.save_audio(path, original, sr)
    loaded, loaded_sr = humecodec.load_audio(path)

    assert loaded_sr == sr
    # Codec may pad frames; compare the overlapping region
    min_frames = min(original.shape[0], loaded.shape[0])
    assert loaded.shape[1] == original.shape[1]
    torch.testing.assert_close(loaded[:min_frames], original[:min_frames], atol=1e-4, rtol=1e-4)


def test_save_load_flac_roundtrip(tmp_path):
    path = str(tmp_path / "test.flac")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)

    humecodec.save_audio(path, original, sr)
    loaded, loaded_sr = humecodec.load_audio(path)

    assert loaded_sr == sr
    # FLAC may pad to block boundaries; compare the overlapping region
    min_frames = min(original.shape[0], loaded.shape[0])
    assert loaded.shape[1] == original.shape[1]
    torch.testing.assert_close(loaded[:min_frames], original[:min_frames], atol=1e-4, rtol=1e-4)


def test_save_load_mp3_roundtrip(tmp_path):
    path = str(tmp_path / "test.mp3")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=1.0)

    humecodec.save_audio(path, original, sr)
    loaded, loaded_sr = humecodec.load_audio(path)

    assert loaded_sr == sr
    # MP3 pads frames, so loaded may be slightly longer; just check it's close
    min_frames = min(original.shape[0], loaded.shape[0])
    assert min_frames > 0
    assert loaded.shape[1] == original.shape[1]
    # Lossy: allow generous tolerance
    diff = (loaded[:min_frames] - original[:min_frames]).abs().mean()
    assert diff < 0.05, f"Mean absolute difference too large: {diff}"


def test_save_load_opus_roundtrip(tmp_path):
    path = str(tmp_path / "test.opus")
    sr = 48000  # Opus native rate
    original = _make_sine(sample_rate=sr, duration=1.0)

    humecodec.save_audio(path, original, sr)
    loaded, loaded_sr = humecodec.load_audio(path)

    assert loaded_sr == sr
    min_frames = min(original.shape[0], loaded.shape[0])
    assert min_frames > 0
    assert loaded.shape[1] == original.shape[1]
    diff = (loaded[:min_frames] - original[:min_frames]).abs().mean()
    assert diff < 0.05, f"Mean absolute difference too large: {diff}"


def test_audio_info(tmp_path):
    path = str(tmp_path / "info_test.wav")
    sr = 16000
    num_channels = 2
    duration = 0.5
    original = _make_sine(sample_rate=sr, duration=duration, num_channels=num_channels)

    humecodec.save_audio(path, original, sr)
    ai = humecodec.info(path)

    assert ai.sample_rate == sr
    assert ai.num_channels == num_channels
    # num_frames may be 0 if the container doesn't store duration in the header
    if ai.num_frames > 0:
        assert ai.num_frames == original.shape[0]


def test_encoder_decoder_streaming(tmp_path):
    path = str(tmp_path / "stream.wav")
    sr = 16000
    num_channels = 1
    chunk_frames = 1024
    num_chunks = 4

    # Encode chunks
    enc = humecodec.MediaEncoder(path)
    enc.add_audio_stream(sample_rate=sr, num_channels=num_channels)
    enc.open()
    for i in range(num_chunks):
        chunk = _make_sine(sample_rate=sr,
                           duration=chunk_frames / sr,
                           freq=440.0,
                           num_channels=num_channels)
        enc.write_audio_chunk(0, chunk)
    enc.flush()
    enc.close()

    # Decode back
    decoder = humecodec.MediaDecoder(path)
    decoder.add_audio_stream(frames_per_chunk=-1)
    decoder.process_all_packets()
    (result,) = decoder.pop_chunks()

    expected_frames = chunk_frames * num_chunks
    assert result is not None
    # Codec may pad; just verify we got at least the expected number of frames
    assert result.shape[0] >= expected_frames
    assert result.shape[1] == num_channels


def test_get_versions():
    versions = humecodec.get_versions()
    expected_keys = {"libavutil", "libavcodec", "libavformat",
                     "libavfilter", "libavdevice"}
    assert expected_keys.issubset(set(versions.keys()))
    for name, ver in versions.items():
        assert len(ver) == 3, f"{name} version should be a 3-tuple"
        assert all(isinstance(v, int) for v in ver)
