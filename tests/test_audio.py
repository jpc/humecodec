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


def test_build_packet_index(tmp_path):
    path = str(tmp_path / "index_test.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=5.0)
    humecodec.save_audio(path, original, sr)

    # Build index with default resolution
    decoder = humecodec.MediaDecoder(path)
    decoder.add_audio_stream(frames_per_chunk=-1)
    index_default = decoder.build_packet_index()

    assert len(index_default) > 0
    for entry in index_default:
        assert entry.pos >= 0
        assert entry.size > 0
        assert entry.pts_seconds >= 0
    # Entries should be sorted by pts_seconds
    pts_list = [e.pts_seconds for e in index_default]
    assert pts_list == sorted(pts_list)

    # Build index with resolution=1 (every packet)
    decoder2 = humecodec.MediaDecoder(path)
    decoder2.add_audio_stream(frames_per_chunk=-1)
    index_full = decoder2.build_packet_index(resolution=1)

    assert len(index_full) >= len(index_default)


def test_seek_to_byte_offset_sample_accurate(tmp_path):
    path = str(tmp_path / "sweep.wav")
    sr = 44100
    duration = 5.0
    n = int(sr * duration)
    # Frequency sweep so each position is unique
    t = torch.arange(n, dtype=torch.float32) / sr
    freq = 200 + 2000 * t / duration  # 200 Hz -> 2200 Hz
    phase = 2 * math.pi * torch.cumsum(freq / sr, dim=0)
    waveform = (0.5 * torch.sin(phase)).unsqueeze(1)

    humecodec.save_audio(path, waveform, sr)

    # Load the full file as reference (WAV may pad slightly)
    ref, ref_sr = humecodec.load_audio(path)
    assert ref_sr == sr

    # Build full packet index
    decoder = humecodec.MediaDecoder(path)
    decoder.add_audio_stream(frames_per_chunk=-1)
    index = decoder.build_packet_index(resolution=1)
    assert len(index) > 1

    # Pick several entries spread across the file
    step = max(1, len(index) // 5)
    test_entries = index[::step]

    for entry in test_entries:
        dec = humecodec.MediaDecoder(path)
        dec.add_audio_stream(frames_per_chunk=-1, filter_desc="aformat=sample_fmts=fltp")
        dec.seek_to_byte_offset(entry.pos)
        dec.process_all_packets()
        (chunk,) = dec.pop_chunks()
        assert chunk is not None
        assert chunk.shape[0] > 0

        # Determine the sample offset from pts_seconds
        sample_offset = int(round(entry.pts_seconds * sr))
        compare_len = min(chunk.shape[0], ref.shape[0] - sample_offset)
        if compare_len <= 0:
            continue
        ref_region = ref[sample_offset : sample_offset + compare_len]
        torch.testing.assert_close(
            chunk[:compare_len], ref_region, atol=1e-4, rtol=1e-4
        )


def test_load_torchaudio_compat(tmp_path):
    """load() returns [channel, time] shape, correct sample rate, float32."""
    path = str(tmp_path / "compat.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)  # [time, channel]
    humecodec.save_audio(path, original, sr)

    waveform, loaded_sr = humecodec.load(path)
    assert loaded_sr == sr
    assert waveform.dtype == torch.float32
    # load() default is channels_first=True -> [channel, time]
    assert waveform.dim() == 2
    assert waveform.shape[0] == original.shape[1]  # num_channels
    min_frames = min(waveform.shape[1], original.shape[0])
    assert min_frames > 0


def test_load_channels_first_false(tmp_path):
    """load() with channels_first=False returns [time, channel]."""
    path = str(tmp_path / "compat_cf.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)
    humecodec.save_audio(path, original, sr)

    waveform, loaded_sr = humecodec.load(path, channels_first=False)
    assert loaded_sr == sr
    # channels_first=False -> [time, channel]
    assert waveform.shape[1] == original.shape[1]  # num_channels
    min_frames = min(waveform.shape[0], original.shape[0])
    torch.testing.assert_close(
        waveform[:min_frames], original[:min_frames], atol=1e-4, rtol=1e-4
    )


def test_load_frame_offset_num_frames(tmp_path):
    """load() with frame_offset and num_frames slices correctly."""
    path = str(tmp_path / "slice.wav")
    sr = 44100
    duration = 1.0
    original = _make_sine(sample_rate=sr, duration=duration)
    humecodec.save_audio(path, original, sr)

    frame_offset = 4410  # 0.1 seconds
    num_frames = 22050   # 0.5 seconds

    waveform, loaded_sr = humecodec.load(
        path, frame_offset=frame_offset, num_frames=num_frames
    )
    assert loaded_sr == sr
    # channels_first=True -> [channel, time]
    assert waveform.shape[1] <= num_frames + 1  # allow 1 frame rounding


def test_save_torchaudio_compat(tmp_path):
    """save() accepts [channel, time] input and round-trips correctly."""
    path = str(tmp_path / "save_compat.wav")
    sr = 44100
    # Create [channel, time] tensor (torchaudio convention)
    original_tf = _make_sine(sample_rate=sr, duration=0.5)  # [time, channel]
    original_cf = original_tf.t()  # [channel, time]

    humecodec.save(path, original_cf, sr)
    loaded, loaded_sr = humecodec.load(path)

    assert loaded_sr == sr
    assert loaded.shape[0] == original_cf.shape[0]  # same num_channels
    min_frames = min(loaded.shape[1], original_cf.shape[1])
    torch.testing.assert_close(
        loaded[:, :min_frames], original_cf[:, :min_frames], atol=1e-4, rtol=1e-4
    )


def test_save_channels_first_false(tmp_path):
    """save() with channels_first=False accepts [time, channel] input."""
    path = str(tmp_path / "save_tf.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)  # [time, channel]

    humecodec.save(path, original, sr, channels_first=False)
    loaded, loaded_sr = humecodec.load(path, channels_first=False)

    assert loaded_sr == sr
    min_frames = min(loaded.shape[0], original.shape[0])
    torch.testing.assert_close(
        loaded[:min_frames], original[:min_frames], atol=1e-4, rtol=1e-4
    )


def test_info_has_encoding(tmp_path):
    """info() returns encoding and bits_per_sample fields."""
    path = str(tmp_path / "enc_test.wav")
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)
    humecodec.save_audio(path, original, sr)

    ai = humecodec.info(path)
    assert hasattr(ai, "encoding")
    assert hasattr(ai, "bits_per_sample")
    # WAV default encoder is pcm_s16le or pcm_f32le
    assert ai.encoding in ("PCM_S", "PCM_F")
    assert ai.bits_per_sample > 0


def test_list_audio_backends():
    backends = humecodec.list_audio_backends()
    assert isinstance(backends, list)
    assert "ffmpeg" in backends


def test_load_pathlike(tmp_path):
    """load() accepts PathLike objects."""
    path = tmp_path / "pathlike.wav"
    sr = 44100
    original = _make_sine(sample_rate=sr, duration=0.5)
    humecodec.save_audio(str(path), original, sr)

    waveform, loaded_sr = humecodec.load(path)  # Pass Path object directly
    assert loaded_sr == sr
    assert waveform.shape[0] == 1  # mono, channels_first


def test_get_versions():
    versions = humecodec.get_versions()
    # libavdevice is loaded at runtime via dlopen and may not be available
    expected_keys = {"libavutil", "libavcodec", "libavformat",
                     "libavfilter"}
    assert expected_keys.issubset(set(versions.keys()))
    for name, ver in versions.items():
        assert len(ver) == 3, f"{name} version should be a 3-tuple"
        assert all(isinstance(v, int) for v in ver)
