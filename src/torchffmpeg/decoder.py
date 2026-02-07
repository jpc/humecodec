from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union

import torch
from torch.utils._pytree import tree_map

from torchffmpeg._extension import get_ext


@dataclass
class SourceStream:
    """The metadata of a source stream."""

    media_type: str
    """The type of the stream."""
    codec: str
    """Short name of the codec."""
    codec_long_name: str
    """Detailed name of the codec."""
    format: Optional[str]
    """Media format."""
    bit_rate: Optional[int]
    """Bit rate of the stream in bits-per-second."""
    num_frames: Optional[int]
    """The number of frames in the stream."""
    bits_per_sample: Optional[int]
    """The number of valid bits in each output sample."""
    metadata: Dict[str, str]
    """Metadata attached to the source stream."""
    time_base: Tuple[int, int]
    """Stream time base as (numerator, denominator). PTS values are in these units."""


@dataclass
class SourceAudioStream(SourceStream):
    """The metadata of an audio source stream."""

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class SourceVideoStream(SourceStream):
    """The metadata of a video source stream."""

    width: int
    """Width of the video frame in pixel."""
    height: int
    """Height of the video frame in pixel."""
    frame_rate: float
    """Frame rate."""


def _parse_si(i):
    media_type = i.media_type
    tb = (i.time_base_num, i.time_base_den)
    if media_type == "audio":
        return SourceAudioStream(
            media_type=i.media_type,
            codec=i.codec_name,
            codec_long_name=i.codec_long_name,
            format=i.format,
            bit_rate=i.bit_rate,
            num_frames=i.num_frames,
            bits_per_sample=i.bits_per_sample,
            metadata=i.metadata,
            time_base=tb,
            sample_rate=i.sample_rate,
            num_channels=i.num_channels,
        )
    if media_type == "video":
        return SourceVideoStream(
            media_type=i.media_type,
            codec=i.codec_name,
            codec_long_name=i.codec_long_name,
            format=i.format,
            bit_rate=i.bit_rate,
            num_frames=i.num_frames,
            bits_per_sample=i.bits_per_sample,
            metadata=i.metadata,
            time_base=tb,
            width=i.width,
            height=i.height,
            frame_rate=i.frame_rate,
        )
    return SourceStream(
        media_type=i.media_type,
        codec=i.codec_name,
        codec_long_name=i.codec_long_name,
        format=None,
        bit_rate=None,
        num_frames=None,
        bits_per_sample=None,
        metadata=i.metadata,
        time_base=tb,
    )


@dataclass
class OutputStream:
    """Output stream configured on MediaDecoder."""

    source_index: int
    """Index of the source stream that this output stream is connected."""
    filter_description: str
    """Description of filter graph applied to the source stream."""
    media_type: str
    """The type of the stream. ``"audio"`` or ``"video"``."""
    format: str
    """Media format."""


@dataclass
class OutputAudioStream(OutputStream):
    """Information about an audio output stream."""

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class OutputVideoStream(OutputStream):
    """Information about a video output stream."""

    width: int
    """Width of the video frame in pixel."""
    height: int
    """Height of the video frame in pixel."""
    frame_rate: float
    """Frame rate."""


def _parse_oi(i):
    media_type = i.media_type
    if media_type == "audio":
        return OutputAudioStream(
            source_index=i.source_index,
            filter_description=i.filter_description,
            media_type=i.media_type,
            format=i.format,
            sample_rate=i.sample_rate,
            num_channels=i.num_channels,
        )
    if media_type == "video":
        return OutputVideoStream(
            source_index=i.source_index,
            filter_description=i.filter_description,
            media_type=i.media_type,
            format=i.format,
            width=i.width,
            height=i.height,
            frame_rate=i.frame_rate,
        )
    raise ValueError(f"Unexpected media_type: {i.media_type}({i})")


def _get_afilter_desc(sample_rate, fmt, num_channels):
    descs = []
    if sample_rate is not None:
        descs.append(f"aresample={sample_rate}")
    if fmt is not None or num_channels is not None:
        parts = []
        if fmt is not None:
            parts.append(f"sample_fmts={fmt}")
        if num_channels is not None:
            parts.append(f"channel_layouts={num_channels}c")
        descs.append(f"aformat={':'.join(parts)}")
    return ",".join(descs) if descs else None


def _get_vfilter_desc(frame_rate, width, height, fmt):
    descs = []
    if frame_rate is not None:
        descs.append(f"fps={frame_rate}")
    scales = []
    if width is not None:
        scales.append(f"width={width}")
    if height is not None:
        scales.append(f"height={height}")
    if scales:
        descs.append(f"scale={':'.join(scales)}")
    if fmt is not None:
        descs.append(f"format=pix_fmts={fmt}")
    return ",".join(descs) if descs else None


class ChunkTensorBase(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, _elem, *_):
        return super().__new__(cls, _elem)

    @classmethod
    def __torch_dispatch__(cls, func, _, args=(), kwargs=None):
        def unwrap(t):
            return t._elem if isinstance(t, cls) else t

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


@dataclass
class Chunk(ChunkTensorBase):
    """Decoded media frames with metadata.

    The instance of this class represents the decoded video/audio frames with
    metadata, and the instance itself behaves like :py:class:`~torch.Tensor`.
    """

    _elem: torch.Tensor

    pts: float
    """Presentation time stamp of the first frame in the chunk. Unit: second."""


InputStreamTypes = TypeVar("InputStream", bound=SourceStream)
OutputStreamTypes = TypeVar("OutputStream", bound=OutputStream)


class MediaDecoder:
    """Fetch and decode audio/video streams chunk by chunk.

    Args:
        src (str, path-like, bytes or file-like object): The media source.
        format (str or None, optional): Override the input format.
            Default: ``None``.
        option (dict of str to str, optional): Custom option passed when
            initializing format context. Default: ``None``.
        buffer_size (int): The internal buffer size in byte. Used only when
            `src` is file-like object. Default: ``4096``.
    """

    def __init__(
        self,
        src: Union[str, Path, BinaryIO],
        format: Optional[str] = None,
        option: Optional[Dict[str, str]] = None,
        buffer_size: int = 4096,
    ):
        ext = get_ext()
        self.src = src
        if isinstance(src, bytes):
            self._be = ext.StreamingMediaDecoderBytes(src, format, option, buffer_size)
        elif hasattr(src, "read"):
            self._be = ext.StreamingMediaDecoderFileObj(src, format, option, buffer_size)
        else:
            self._be = ext.StreamingMediaDecoder(os.path.normpath(src), format, option)

        i = self._be.find_best_audio_stream()
        self._default_audio_stream = None if i < 0 else i
        i = self._be.find_best_video_stream()
        self._default_video_stream = None if i < 0 else i

    @property
    def num_src_streams(self):
        """Number of streams found in the provided media source."""
        return self._be.num_src_streams()

    @property
    def num_out_streams(self):
        """Number of output streams configured by client code."""
        return self._be.num_out_streams()

    @property
    def default_audio_stream(self):
        """The index of default audio stream. ``None`` if there is no audio stream."""
        return self._default_audio_stream

    @property
    def default_video_stream(self):
        """The index of default video stream. ``None`` if there is no video stream."""
        return self._default_video_stream

    def get_metadata(self) -> Dict[str, str]:
        """Get the metadata of the source media."""
        return self._be.get_metadata()

    def get_src_stream_info(self, i: int) -> InputStreamTypes:
        """Get the metadata of source stream."""
        return _parse_si(self._be.get_src_stream_info(i))

    def get_out_stream_info(self, i: int) -> OutputStreamTypes:
        """Get the metadata of output stream."""
        info = self._be.get_out_stream_info(i)
        return _parse_oi(info)

    def seek(self, timestamp: float, mode: str = "precise"):
        """Seek the stream to the given timestamp [second].

        Args:
            timestamp (float): Target time in second.
            mode (str): Controls how seek is done.
                Valid choices are ``"key"``, ``"any"``, ``"precise"``.
        """
        modes = {
            "key": 0,
            "any": 1,
            "precise": 2,
        }
        if mode not in modes:
            raise ValueError(f"The value of mode must be one of {list(modes.keys())}. Found: {mode}")
        self._be.seek(timestamp, modes[mode])

    def build_packet_index(self, stream_index: Optional[int] = None):
        """Scan all packets for a stream and return a lightweight seek index.

        Each entry contains ``pts_seconds``, ``pos`` (byte offset), ``size``,
        and ``is_key``.  The byte offsets can be passed to
        :meth:`seek_to_byte_offset` for instantaneous seeking without any
        container-level scanning.

        After this call the read position is at EOF; call :meth:`seek` or
        :meth:`seek_to_byte_offset` before continuing to decode.

        Args:
            stream_index (int or None, optional): Source stream to index.
                Defaults to the best audio stream.
        """
        if stream_index is None:
            stream_index = self._default_audio_stream
        if stream_index is None:
            raise RuntimeError("No stream to index (no audio stream found and no stream_index given).")
        return self._be.build_packet_index(stream_index)

    def seek_to_byte_offset(self, offset: int):
        """Seek directly to a byte position in the underlying file.

        This is intended for use with offsets obtained from
        :meth:`build_packet_index` and skips all container-level parsing.

        Args:
            offset (int): Byte offset (from :attr:`PacketIndexEntry.pos`).
        """
        self._be.seek_to_byte_offset(offset)

    def add_basic_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        format: Optional[str] = "fltp",
        sample_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
    ):
        """Add output audio stream with basic options."""
        self.add_audio_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index=stream_index,
            decoder=decoder,
            decoder_option=decoder_option,
            filter_desc=_get_afilter_desc(sample_rate, format, num_channels),
        )

    def add_basic_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        format: Optional[str] = "rgb24",
        frame_rate: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        hw_accel: Optional[str] = None,
    ):
        """Add output video stream with basic options."""
        self.add_video_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index=stream_index,
            decoder=decoder,
            decoder_option=decoder_option,
            filter_desc=_get_vfilter_desc(frame_rate, width, height, format),
            hw_accel=hw_accel,
        )

    def add_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add output audio stream.

        Args:
            frames_per_chunk (int): Number of frames returned as one chunk.
            buffer_chunk_size (int, optional): Internal buffer size. Default: ``3``.
            stream_index (int or None, optional): The source audio stream index.
            decoder (str or None, optional): The name of the decoder.
            decoder_option (dict or None, optional): Options passed to decoder.
            filter_desc (str or None, optional): Filter description.
        """
        i = self.default_audio_stream if stream_index is None else stream_index
        if i is None:
            raise RuntimeError("There is no audio stream.")
        self._be.add_audio_stream(
            i,
            frames_per_chunk,
            buffer_chunk_size,
            filter_desc,
            decoder,
            decoder_option or {},
        )

    def add_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        filter_desc: Optional[str] = None,
        hw_accel: Optional[str] = None,
    ):
        """Add output video stream.

        Args:
            frames_per_chunk (int): Number of frames returned as one chunk.
            buffer_chunk_size (int, optional): Internal buffer size. Default: ``3``.
            stream_index (int or None, optional): The source video stream index.
            decoder (str or None, optional): The name of the decoder.
            decoder_option (dict or None, optional): Options passed to decoder.
            filter_desc (str or None, optional): Filter description.
            hw_accel (str or None, optional): Enable hardware acceleration.
        """
        i = self.default_video_stream if stream_index is None else stream_index
        if i is None:
            raise RuntimeError("There is no video stream.")
        self._be.add_video_stream(
            i,
            frames_per_chunk,
            buffer_chunk_size,
            filter_desc,
            decoder,
            decoder_option or {},
            hw_accel,
        )

    def remove_stream(self, i: int):
        """Remove an output stream."""
        self._be.remove_stream(i)

    def process_packet(self, timeout: Optional[float] = None, backoff: float = 10.0) -> int:
        """Read the source media and process one packet.

        Returns:
            int: ``0`` if a packet was processed, ``1`` if EOF was reached.
        """
        return self._be.process_packet(timeout, backoff)

    def process_all_packets(self):
        """Process packets until EOF."""
        self._be.process_all_packets()

    def is_buffer_ready(self) -> bool:
        """Returns true if all output streams have at least one chunk filled."""
        return self._be.is_buffer_ready()

    def pop_chunks(self) -> Tuple[Optional[Chunk]]:
        """Pop one chunk from all the output stream buffers.

        Returns decoded frames as Chunk objects, which wrap torch.Tensor instances.
        The tensors are transferred via DLPack protocol for efficient zero-copy exchange.
        """
        ret = []
        # Use DLPack-based pop_chunks for efficient tensor transfer
        for item in self._be.pop_chunks_dlpack():
            if item is None:
                ret.append(None)
            else:
                # item is a tuple (capsule, pts) for DLPack transfer
                capsule, pts = item
                # Convert DLPack capsule to torch.Tensor
                tensor = torch.from_dlpack(capsule)
                ret.append(Chunk(tensor, pts))
        return ret

    def fill_buffer(self, timeout: Optional[float] = None, backoff: float = 10.0) -> int:
        """Keep processing packets until all buffers have at least one chunk.

        Returns:
            int: ``0`` if buffers are ready, ``1`` if EOF was reached.
        """
        return self._be.fill_buffer(timeout, backoff)

    def stream(
        self, timeout: Optional[float] = None, backoff: float = 10.0
    ) -> Iterator[Tuple[Optional[Chunk], ...]]:
        """Return an iterator that generates output tensors."""
        if self.num_out_streams == 0:
            raise RuntimeError("No output stream is configured.")

        while True:
            if self.fill_buffer(timeout, backoff):
                break
            yield self.pop_chunks()

        while True:
            chunks = self.pop_chunks()
            if all(c is None for c in chunks):
                return
            yield chunks
