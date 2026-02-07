from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union

import torch

from torchffmpeg._extension import get_ext


@dataclass
class CodecConfig:
    """Codec configuration."""

    bit_rate: int = -1
    """Bit rate."""

    compression_level: int = -1
    """Compression level."""

    qscale: Optional[int] = None
    """Global quality factor. Enables variable bit rate."""

    gop_size: int = -1
    """The number of pictures in a group of pictures, or 0 for intra_only."""

    max_b_frames: int = -1
    """Maximum number of B-frames between non-B-frames."""


def _convert_config(cfg: Optional[CodecConfig]):
    if cfg is None:
        return None
    ext = get_ext()
    return ext.CodecConfig(
        cfg.bit_rate,
        cfg.compression_level,
        cfg.qscale,
        cfg.gop_size,
        cfg.max_b_frames,
    )


class MediaEncoder:
    """Encode and write audio/video streams chunk by chunk.

    Args:
        dst (str, path-like or file-like object): The destination where
            encoded data are written.
        format (str or None, optional): Override the output format.
            Default: ``None``.
        buffer_size (int): The internal buffer size in byte. Used only when
            `dst` is a file-like object. Default: ``4096``.
    """

    def __init__(
        self,
        dst: Union[str, Path, BinaryIO],
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ):
        ext = get_ext()
        if hasattr(dst, "write"):
            self._s = ext.StreamingMediaEncoderFileObj(dst, format, buffer_size)
        else:
            self._s = ext.StreamingMediaEncoder(str(dst), format)
        self._is_open = False

    def add_audio_stream(
        self,
        sample_rate: int,
        num_channels: int,
        format: str = "flt",
        *,
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_sample_rate: Optional[int] = None,
        encoder_num_channels: Optional[int] = None,
        encoder_format: Optional[str] = None,
        codec_config: Optional[CodecConfig] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add an output audio stream.

        Args:
            sample_rate (int): The sample rate.
            num_channels (int): The number of channels.
            format (str, optional): Input sample format. Default: ``"flt"``.
            encoder (str or None, optional): The encoder name. Default: ``None``.
            encoder_option (dict or None, optional): Options for encoder.
            encoder_sample_rate (int or None, optional): Override encoding sample rate.
            encoder_num_channels (int or None, optional): Override encoding channels.
            encoder_format (str or None, optional): Override encoding format.
            codec_config (CodecConfig or None, optional): Codec configuration.
            filter_desc (str or None, optional): Additional filter processing.
        """
        self._s.add_audio_stream(
            sample_rate,
            num_channels,
            format,
            encoder,
            encoder_option,
            encoder_format,
            encoder_sample_rate,
            encoder_num_channels,
            _convert_config(codec_config),
            filter_desc,
        )

    def add_video_stream(
        self,
        frame_rate: float,
        width: int,
        height: int,
        format: str = "rgb24",
        *,
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_frame_rate: Optional[float] = None,
        encoder_width: Optional[int] = None,
        encoder_height: Optional[int] = None,
        encoder_format: Optional[str] = None,
        codec_config: Optional[CodecConfig] = None,
        filter_desc: Optional[str] = None,
        hw_accel: Optional[str] = None,
    ):
        """Add an output video stream.

        Args:
            frame_rate (float): Frame rate of the video.
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            format (str, optional): Input pixel format. Default: ``"rgb24"``.
            encoder (str or None, optional): The encoder name.
            encoder_option (dict or None, optional): Options for encoder.
            encoder_frame_rate (float or None, optional): Override encoding frame rate.
            encoder_width (int or None, optional): Override encoding width.
            encoder_height (int or None, optional): Override encoding height.
            encoder_format (str or None, optional): Override encoding format.
            codec_config (CodecConfig or None, optional): Codec configuration.
            filter_desc (str or None, optional): Additional filter processing.
            hw_accel (str or None, optional): Enable hardware acceleration.
        """
        self._s.add_video_stream(
            frame_rate,
            width,
            height,
            format,
            encoder,
            encoder_option,
            encoder_format,
            encoder_frame_rate,
            encoder_width,
            encoder_height,
            hw_accel,
            _convert_config(codec_config),
            filter_desc,
        )

    def set_metadata(self, metadata: Dict[str, str]):
        """Set file-level metadata."""
        self._s.set_metadata(metadata)

    def open(self, option: Optional[Dict[str, str]] = None) -> "MediaEncoder":
        """Open the output file / device and write the header.

        Returns self for use as a context manager.
        """
        if not self._is_open:
            self._s.open(option)
            self._is_open = True
        return self

    def close(self):
        """Close the output."""
        if self._is_open:
            self._s.close()
            self._is_open = False

    def write_audio_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float] = None):
        """Write audio data.

        Args:
            i (int): Stream index.
            chunk (Tensor): Waveform tensor. Shape: ``(frame, channel)``.
            pts (float or None, optional): Presentation timestamp override.

        Note:
            The tensor is transferred via DLPack protocol for efficient zero-copy exchange.
        """
        # Use DLPack-based method for efficient tensor transfer
        capsule = chunk.__dlpack__()
        self._s.write_audio_chunk_dlpack(i, capsule, pts)

    def write_video_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float] = None):
        """Write video/image data.

        Args:
            i (int): Stream index.
            chunk (Tensor): Video tensor. Shape: ``(time, channel, height, width)``.
            pts (float or None, optional): Presentation timestamp override.

        Note:
            The tensor is transferred via DLPack protocol for efficient zero-copy exchange.
        """
        # Use DLPack-based method for efficient tensor transfer
        capsule = chunk.__dlpack__()
        self._s.write_video_chunk_dlpack(i, capsule, pts)

    def flush(self):
        """Flush the frames from encoders and write to the destination."""
        self._s.flush()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.flush()
        self.close()
