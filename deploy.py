"""Modal deploy entry for ffmpeg.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import modal
from tongflow import deploy
from tongflow.protocol import (
    asset_as_path,
    asset_from_path,
    prompt_media_to_bytes,
)

logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "tongflow==0.1.0",
        "moviepy",
    )
)
app = modal.App(Path(__file__).resolve().parent.name, image=image)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

with image.imports():
    import subprocess

    from moviepy import (
        AudioFileClip,
        VideoFileClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )


# ========== Video processing helpers ==========
def concat_videos_helper(
    local_paths: list[Path],
    output_path: Path,
    quality: str = "medium",
    resolution_scale: float = 1.0,
    fps_limit: Optional[int] = None,
    optimize_memory: bool = False,
):
    """Concatenate videos with resource management and overall performance tuning."""
    logger.info(
        f"start concatenating videos: {len(local_paths)} files, quality={quality}, "
        f"resolution_scale={resolution_scale}, fps_limit={fps_limit}"
    )
    start_time = time.time()
    clips: list = []
    try:
        for path in local_paths:
            clip = VideoFileClip(str(path))
            if resolution_scale != 1.0:
                clip = clip.resized(resolution_scale)
            if fps_limit and clip.fps > fps_limit:
                clip = clip.with_fps(fps_limit)
            clips.append(clip)
        if optimize_memory:
            final = concatenate_videoclips(clips, method="chain")
        else:
            final = concatenate_videoclips(clips, method="compose")
        ffmpeg_params = ["-threads", "0"]
        if quality == "fast":
            preset = "ultrafast"
            crf = "28"
            ffmpeg_params.extend(
                [
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "fastdecode",
                    "-x264-params",
                    "bframes=0:b-adapt=0:no-scenecut",
                ]
            )
        elif quality == "medium":
            preset = "fast"
            crf = "23"
        else:
            preset = "medium"
            crf = "20"
        ffmpeg_params.extend(["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset=preset,
            ffmpeg_params=["-crf", crf] + ffmpeg_params,
        )
        final.close()
        logger.info(f"video concat done, total elapsed {time.time() - start_time:.2f}s")
    finally:
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass


def concat_videos_fast_copy(local_paths: list[Path], output_path: Path) -> None:
    """Ultra-fast video concat: ffmpeg concat demuxer + copy (requires identical codec)."""
    file_list_path = output_path.parent / "filelist.txt"
    with open(file_list_path, "w", encoding="utf-8") as f:
        for path in local_paths:
            escaped = str(path).replace("'", "'\"'\"'")
            f.write(f"file '{escaped}'\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
    )


def remove_video_audio_helper(local_path: Path, out_video: Path) -> None:
    """Strip audio track; copy video stream (silent output)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(local_path),
            "-an",
            "-vcodec",
            "copy",
            str(out_video),
        ],
        check=True,
    )


def merge_av_helper(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def extract_audio_helper(
    video_path: Path, output_path: Path, audio_format: str = "mp3"
) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn"]
    if audio_format == "mp3":
        cmd.extend(["-acodec", "libmp3lame"])
    elif audio_format == "wav":
        cmd.extend(["-acodec", "pcm_s16le"])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def get_first_frame_helper(video_path: Path, output_path: Path) -> None:
    video = VideoFileClip(str(video_path))
    try:
        video.save_frame(str(output_path), t=0)
    finally:
        video.close()


def get_last_frame_helper(video_path: Path, output_path: Path) -> None:
    video = VideoFileClip(str(video_path))
    try:
        fps = video.fps if video.fps else 24.0
        t = max(0, video.duration - (0.5 / fps))
        video.save_frame(str(output_path), t=t)
    finally:
        video.close()


# ========== ABI bindings ==========
from tongflow.models.concat_videos import ConcatVideosInput, ConcatVideosOutput
from tongflow.models.extract_audio import ExtractAudioInput, ExtractAudioOutput
from tongflow.models.get_first_frame import GetFirstFrameInput, GetFirstFrameOutput
from tongflow.models.get_last_frame import GetLastFrameInput, GetLastFrameOutput
from tongflow.models.merge_video_audio import (
    MergeVideoAudioInput,
    MergeVideoAudioOutput,
)
from tongflow.models.remove_video_audio import (
    RemoveVideoAudioInput,
    RemoveVideoAudioOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot

# Audio format for ExtractAudio is plugin-internal — not part of ABI.
DEFAULT_AUDIO_FORMAT = "mp3"


@deploy
@app.cls(image=image, cpu=0.5, memory=1024, timeout=3600, scaledown_window=5)
class Inference:
    @modal.method()
    @node_slot(NodeSlots.CONCAT_VIDEOS)
    def concat_videos(self, input: ConcatVideosInput) -> ConcatVideosOutput:
        if not input.videos:
            return ConcatVideosOutput(
                success=False, error="Missing `videos` (Asset[])"
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            local_files: list[Path] = []
            for i, v in enumerate(input.videos):
                local = tmp_dir / f"in_{i}.mp4"
                local.write_bytes(prompt_media_to_bytes(v))
                local_files.append(local)
            out = tmp_dir / f"{uuid.uuid4().hex}.mp4"
            try:
                concat_videos_fast_copy(local_files, out)
            except Exception:
                concat_videos_helper(local_files, out)
            return ConcatVideosOutput(success=True, video=asset_from_path(out))

    @modal.method()
    @node_slot(NodeSlots.EXTRACT_AUDIO)
    def extract_audio(self, input: ExtractAudioInput) -> ExtractAudioOutput:
        if input.video is None:
            return ExtractAudioOutput(success=False, error="Missing `video` Asset")
        with asset_as_path(input.video, suffix=".mp4") as in_path:
            out = in_path.with_suffix(f".{DEFAULT_AUDIO_FORMAT}")
            extract_audio_helper(in_path, out, DEFAULT_AUDIO_FORMAT)
            return ExtractAudioOutput(success=True, audio=asset_from_path(out))

    @modal.method()
    @node_slot(NodeSlots.REMOVE_VIDEO_AUDIO)
    def remove_video_audio(
        self, input: RemoveVideoAudioInput
    ) -> RemoveVideoAudioOutput:
        if input.video is None:
            return RemoveVideoAudioOutput(
                success=False, error="Missing `video` Asset"
            )
        with asset_as_path(input.video, suffix=".mp4") as in_path:
            out_v = in_path.with_name(f"{uuid.uuid4().hex}.mp4")
            remove_video_audio_helper(in_path, out_v)
            return RemoveVideoAudioOutput(
                success=True,
                video=asset_from_path(out_v),
            )

    @modal.method()
    @node_slot(NodeSlots.MERGE_VIDEO_AUDIO)
    def merge_video_audio(
        self, input: MergeVideoAudioInput
    ) -> MergeVideoAudioOutput:
        if input.video is None or input.audio is None:
            return MergeVideoAudioOutput(
                success=False, error="Missing `video` / `audio` Asset"
            )
        with asset_as_path(input.video, suffix=".mp4") as v_path, asset_as_path(
            input.audio, suffix=".mp3"
        ) as a_path:
            out = v_path.with_name(f"{uuid.uuid4().hex}.mp4")
            merge_av_helper(v_path, a_path, out)
            return MergeVideoAudioOutput(success=True, video=asset_from_path(out))

    @modal.method()
    @node_slot(NodeSlots.GET_FIRST_FRAME)
    def get_first_frame(self, input: GetFirstFrameInput) -> GetFirstFrameOutput:
        if input.video is None:
            return GetFirstFrameOutput(
                success=False, error="Missing `video` Asset"
            )
        with asset_as_path(input.video, suffix=".mp4") as in_path:
            out = in_path.with_suffix(".png")
            get_first_frame_helper(in_path, out)
            return GetFirstFrameOutput(success=True, image=asset_from_path(out))

    @modal.method()
    @node_slot(NodeSlots.GET_LAST_FRAME)
    def get_last_frame(self, input: GetLastFrameInput) -> GetLastFrameOutput:
        if input.video is None:
            return GetLastFrameOutput(
                success=False, error="Missing `video` Asset"
            )
        with asset_as_path(input.video, suffix=".mp4") as in_path:
            out = in_path.with_suffix(".png")
            get_last_frame_helper(in_path, out)
            return GetLastFrameOutput(success=True, image=asset_from_path(out))
