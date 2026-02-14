"""humecodec - Standalone FFmpeg integration for PyTorch.

Provides MediaDecoder/MediaEncoder classes and convenience functions
for loading/saving audio and video with PyTorch tensors.
"""

from __future__ import annotations

import os
from typing import BinaryIO, Dict, Optional, Tuple, Union

import torch

from humecodec._extension import get_ext
from humecodec.decoder import (
    Chunk,
    MediaDecoder,
    OutputAudioStream,
    OutputStream,
    OutputVideoStream,
    SourceAudioStream,
    SourceStream,
    SourceVideoStream,
)
from humecodec.encoder import CodecConfig, MediaEncoder

__all__ = [
    "MediaDecoder",
    "MediaEncoder",
    "CodecConfig",
    "Chunk",
    "SourceStream",
    "SourceAudioStream",
    "SourceVideoStream",
    "OutputStream",
    "OutputAudioStream",
    "OutputVideoStream",
    "AudioInfo",
    "load",
    "save",
    "load_audio",
    "save_audio",
    "info",
    "list_audio_backends",
    "get_audio_decoders",
    "get_audio_encoders",
    "get_video_decoders",
    "get_video_encoders",
    "get_muxers",
    "get_demuxers",
    "get_versions",
]


class AudioInfo:
    """Metadata about an audio file."""

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        num_frames: int,
        codec: str,
        format: str,
        bits_per_sample: int = 0,
        encoding: str = "",
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.codec = codec
        self.format = format
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding

    def __repr__(self):
        return (
            f"AudioInfo(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"num_frames={self.num_frames}, "
            f"codec='{self.codec}', format='{self.format}', "
            f"bits_per_sample={self.bits_per_sample}, "
            f"encoding='{self.encoding}')"
        )


_ENCODING_TO_ENCODER = {
    ("PCM_S", 16): "pcm_s16le",
    ("PCM_S", 24): "pcm_s24le",
    ("PCM_S", 32): "pcm_s32le",
    ("PCM_U", 8): "pcm_u8",
    ("PCM_F", 32): "pcm_f32le",
    ("PCM_F", 64): "pcm_f64le",
}

_CODEC_TO_ENCODING = {
    "pcm_s16le": ("PCM_S", 16),
    "pcm_s16be": ("PCM_S", 16),
    "pcm_s24le": ("PCM_S", 24),
    "pcm_s24be": ("PCM_S", 24),
    "pcm_s32le": ("PCM_S", 32),
    "pcm_s32be": ("PCM_S", 32),
    "pcm_u8": ("PCM_U", 8),
    "pcm_f32le": ("PCM_F", 32),
    "pcm_f32be": ("PCM_F", 32),
    "pcm_f64le": ("PCM_F", 64),
    "pcm_f64be": ("PCM_F", 64),
    "flac": ("FLAC", 0),
    "mp3": ("MP3", 0),
    "mp3float": ("MP3", 0),
    "vorbis": ("VORBIS", 0),
    "opus": ("OPUS", 0),
    "aac": ("AAC", 0),
    "libopus": ("OPUS", 0),
    "libvorbis": ("VORBIS", 0),
    "libmp3lame": ("MP3", 0),
}


def _encoding_to_encoder(encoding: str, bits_per_sample: int) -> Optional[str]:
    """Map torchaudio encoding name + bits_per_sample to an FFmpeg encoder name."""
    return _ENCODING_TO_ENCODER.get((encoding, bits_per_sample))


def _codec_to_encoding(codec: str) -> Tuple[str, int]:
    """Map FFmpeg codec name to (torchaudio encoding, bits_per_sample)."""
    return _CODEC_TO_ENCODING.get(codec, ("UNKNOWN", 0))


def _resolve_uri(uri: Union[str, os.PathLike, BinaryIO]) -> Union[str, BinaryIO]:
    """Convert a uri to a str path or pass through BinaryIO."""
    if isinstance(uri, (str, bytes)):
        return str(uri)
    if isinstance(uri, os.PathLike):
        return os.fspath(uri)
    # Assume BinaryIO
    return uri


def load(
    uri: Union[str, os.PathLike, BinaryIO],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio from a file (torchaudio-compatible API).

    Args:
        uri: Path to the audio file or a file-like object.
        frame_offset: Number of frames to skip from the start.
        num_frames: Maximum number of frames to read. ``-1`` reads all.
        normalize: If ``True``, return float32 in [-1, 1]. If ``False``,
            return the native sample format.
        channels_first: If ``True``, return shape ``(channel, time)``.
            If ``False``, return shape ``(time, channel)``.
        format: Override the input format.
        buffer_size: Buffer size for file-like objects.
        backend: Accepted but ignored (always uses ffmpeg).

    Returns:
        Tuple of ``(waveform, sample_rate)``.
    """
    src = _resolve_uri(uri)

    # Probe sample rate to convert frame_offset/num_frames to seconds
    offset_sec = 0.0
    duration_sec: Optional[float] = None

    if frame_offset > 0 or num_frames != -1:
        # Need sample rate to convert frames to seconds
        probe_decoder = MediaDecoder(src, format=format, buffer_size=buffer_size)
        if probe_decoder.default_audio_stream is None:
            raise RuntimeError(f"No audio stream found in '{uri}'")
        si = probe_decoder.get_src_stream_info(probe_decoder.default_audio_stream)
        sr = int(si.sample_rate)
        if frame_offset > 0:
            offset_sec = frame_offset / sr
        if num_frames > 0:
            duration_sec = num_frames / sr

    # Build filter
    filter_parts = []
    if normalize:
        filter_parts.append("aformat=sample_fmts=fltp")
    filter_desc = ",".join(filter_parts) if filter_parts else None

    decoder = MediaDecoder(src, format=format, buffer_size=buffer_size)
    decoder.add_audio_stream(
        frames_per_chunk=-1,
        buffer_chunk_size=-1,
        filter_desc=filter_desc,
    )

    if offset_sec > 0:
        decoder.seek(offset_sec)

    decoder.process_all_packets()

    chunks = decoder.pop_chunks()
    if chunks[0] is None:
        raise RuntimeError(f"Failed to decode audio from '{uri}'")

    waveform = chunks[0]._elem if hasattr(chunks[0], "_elem") else torch.Tensor(chunks[0])

    if duration_sec is not None:
        out_info = decoder.get_out_stream_info(0)
        out_sr = int(out_info.sample_rate)
        max_frames = int(duration_sec * out_sr)
        if waveform.size(0) > max_frames:
            waveform = waveform[:max_frames]

    # Get the actual output sample rate
    out_info = decoder.get_out_stream_info(0)
    actual_sr = int(out_info.sample_rate)

    if channels_first:
        # [time, channel] -> [channel, time]
        waveform = waveform.t()

    return waveform, actual_sr


def save(
    uri: Union[str, os.PathLike, BinaryIO],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[Union[CodecConfig, float, int]] = None,
) -> None:
    """Save audio to a file (torchaudio-compatible API).

    Args:
        uri: Path to the output file or a file-like object.
        src: Audio tensor. If ``channels_first=True``, shape is
            ``(channel, time)``; otherwise ``(time, channel)``.
        sample_rate: Sample rate of the waveform.
        channels_first: If ``True``, input has shape ``(channel, time)``.
        format: Override the output format.
        encoding: Encoding to use (e.g. ``"PCM_S"``, ``"PCM_F"``).
        bits_per_sample: Bits per sample for the encoding.
        buffer_size: Buffer size for file-like objects.
        backend: Accepted but ignored (always uses ffmpeg).
        compression: Codec configuration. Can be a :class:`CodecConfig`,
            or a numeric value mapped to the default codec quality option.
    """
    if src.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {src.dim()}D")

    if channels_first:
        # [channel, time] -> [time, channel]
        waveform = src.t().contiguous()
    else:
        waveform = src

    num_channels = waveform.size(1)
    dst = _resolve_uri(uri)

    # Resolve encoder from encoding/bits_per_sample
    encoder_name: Optional[str] = None
    if encoding is not None:
        bps = bits_per_sample if bits_per_sample is not None else 16
        encoder_name = _encoding_to_encoder(encoding, bps)

    # Resolve codec_config from compression
    codec_config: Optional[CodecConfig] = None
    encoder_option: Optional[Dict[str, str]] = None
    if isinstance(compression, CodecConfig):
        codec_config = compression
    elif compression is not None:
        codec_config = CodecConfig(bit_rate=int(compression))

    enc = MediaEncoder(dst, format=format, buffer_size=buffer_size)
    enc.add_audio_stream(
        sample_rate=sample_rate,
        num_channels=num_channels,
        format="flt",
        encoder=encoder_name,
        encoder_option=encoder_option,
        codec_config=codec_config,
    )

    with enc.open():
        enc.write_audio_chunk(0, waveform.float().contiguous())


def list_audio_backends() -> list:
    """List available audio backends.

    Returns:
        A list containing ``"ffmpeg"``.
    """
    return ["ffmpeg"]


def load_audio(
    path: str,
    format: Optional[str] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
    sample_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file.

    Args:
        path: Path to the audio file.
        format: Override the input format. Default: ``None``.
        offset: Start reading from this time (seconds). Default: ``0.0``.
        duration: Maximum duration to read (seconds). Default: ``None`` (read all).
        sample_rate: If provided, resample to this rate.
        num_channels: If provided, remix to this many channels.

    Returns:
        Tuple of (waveform, sample_rate) where waveform has shape ``(num_frames, num_channels)``
        and dtype ``torch.float32``.
    """
    decoder = MediaDecoder(path, format=format)

    # Build filter description for resampling/remixing
    filter_parts = []
    if sample_rate is not None:
        filter_parts.append(f"aresample={sample_rate}")
    filter_parts.append("aformat=sample_fmts=fltp")
    if num_channels is not None:
        filter_parts[-1] += f":channel_layouts={num_channels}c"
    filter_desc = ",".join(filter_parts) if filter_parts else None

    decoder.add_audio_stream(
        frames_per_chunk=-1,
        buffer_chunk_size=-1,
        filter_desc=filter_desc,
    )

    if offset > 0:
        decoder.seek(offset)

    decoder.process_all_packets()

    chunks = decoder.pop_chunks()
    if chunks[0] is None:
        raise RuntimeError(f"Failed to decode audio from '{path}'")

    waveform = chunks[0]._elem if hasattr(chunks[0], "_elem") else torch.Tensor(chunks[0])

    if duration is not None:
        # Determine effective sample rate
        out_info = decoder.get_out_stream_info(0)
        sr = int(out_info.sample_rate)
        max_frames = int(duration * sr)
        if waveform.size(0) > max_frames:
            waveform = waveform[:max_frames]

    # Get the actual output sample rate
    out_info = decoder.get_out_stream_info(0)
    sr = int(out_info.sample_rate)

    return waveform, sr


def save_audio(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int,
    format: Optional[str] = None,
    encoder: Optional[str] = None,
    encoder_option: Optional[Dict[str, str]] = None,
    codec_config: Optional[CodecConfig] = None,
):
    """Save a waveform tensor to an audio file.

    Args:
        path: Destination path.
        waveform: Audio tensor. Shape: ``(num_frames, num_channels)``.
        sample_rate: Sample rate of the waveform.
        format: Override the output format. Default: ``None``.
        encoder: The encoder name. Default: ``None``.
        encoder_option: Options for encoder. Default: ``None``.
        codec_config: Codec configuration. Default: ``None``.
    """
    if waveform.dim() != 2:
        raise ValueError(f"Expected 2D tensor (frames, channels), got {waveform.dim()}D")

    num_channels = waveform.size(1)

    enc = MediaEncoder(path, format=format)
    enc.add_audio_stream(
        sample_rate=sample_rate,
        num_channels=num_channels,
        format="flt",
        encoder=encoder,
        encoder_option=encoder_option,
        codec_config=codec_config,
    )

    with enc.open():
        enc.write_audio_chunk(0, waveform.float().contiguous())


def info(path: str, format: Optional[str] = None) -> AudioInfo:
    """Get audio metadata without decoding.

    Args:
        path: Path to the audio file.
        format: Override the input format. Default: ``None``.

    Returns:
        AudioInfo with sample_rate, num_channels, num_frames, codec, format.
    """
    decoder = MediaDecoder(path, format=format)
    if decoder.default_audio_stream is None:
        raise RuntimeError(f"No audio stream found in '{path}'")

    si = decoder.get_src_stream_info(decoder.default_audio_stream)
    enc, bps = _codec_to_encoding(si.codec)
    return AudioInfo(
        sample_rate=int(si.sample_rate),
        num_channels=si.num_channels,
        num_frames=si.num_frames if si.num_frames else 0,
        codec=si.codec,
        format=si.format or "",
        bits_per_sample=bps,
        encoding=enc,
    )


# FFmpeg query functions
def get_audio_decoders() -> Dict[str, str]:
    """List available audio decoders."""
    return get_ext().get_audio_decoders()


def get_audio_encoders() -> Dict[str, str]:
    """List available audio encoders."""
    return get_ext().get_audio_encoders()


def get_video_decoders() -> Dict[str, str]:
    """List available video decoders."""
    return get_ext().get_video_decoders()


def get_video_encoders() -> Dict[str, str]:
    """List available video encoders."""
    return get_ext().get_video_encoders()


def get_muxers() -> Dict[str, str]:
    """List available muxers (output formats)."""
    return get_ext().get_muxers()


def get_demuxers() -> Dict[str, str]:
    """List available demuxers (input formats)."""
    return get_ext().get_demuxers()


def get_versions() -> Dict[str, Tuple[int, int, int]]:
    """Get FFmpeg library versions."""
    return get_ext().get_versions()
