"""torchffmpeg - Standalone FFmpeg integration for PyTorch.

Provides MediaDecoder/MediaEncoder classes and convenience functions
for loading/saving audio and video with PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from torchffmpeg._extension import get_ext
from torchffmpeg.decoder import (
    Chunk,
    MediaDecoder,
    OutputAudioStream,
    OutputStream,
    OutputVideoStream,
    SourceAudioStream,
    SourceStream,
    SourceVideoStream,
)
from torchffmpeg.encoder import CodecConfig, MediaEncoder

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
    "load_audio",
    "save_audio",
    "info",
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

    def __init__(self, sample_rate: int, num_channels: int, num_frames: int, codec: str, format: str):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.codec = codec
        self.format = format

    def __repr__(self):
        return (
            f"AudioInfo(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"num_frames={self.num_frames}, "
            f"codec='{self.codec}', format='{self.format}')"
        )


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
    return AudioInfo(
        sample_rate=int(si.sample_rate),
        num_channels=si.num_channels,
        num_frames=si.num_frames if si.num_frames else 0,
        codec=si.codec,
        format=si.format or "",
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
