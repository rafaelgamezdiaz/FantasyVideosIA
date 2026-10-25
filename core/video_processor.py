import os
import sys
import time
import json
import shutil
import subprocess
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple

from core.engine import InferenceEngine

class VideoMetadata:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
        duration: float,
        has_audio: bool
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.duration = duration
        self.has_audio = has_audio

class VideoProcessor:
    """
    High-performance, memory-efficient video restoration and upscaling engine.
    Uses FFmpeg stdin/stdout rawvideo pipe streaming to maintain minimal RAM usage
    and leverages Intel QuickSync (QSV) or MediaFoundation hardware encoders.
    """
    def __init__(self, ffmpeg_bin: str, engine: InferenceEngine):
        self.ffmpeg_bin = ffmpeg_bin
        self.engine = engine
        if not self.ffmpeg_bin or not os.path.exists(self.ffmpeg_bin):
            raise FileNotFoundError(f"FFmpeg executable not found at: {self.ffmpeg_bin}")

    def probe_video(self, video_path: str) -> VideoMetadata:
        """
        Extracts video stream dimensions, framerate, frame count, and audio status.
        """
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-i", video_path
        ]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, errors="replace")
        stderr = res.stderr

        width, height = 1920, 1080
        fps = 30.0
        duration = 0.0
        has_audio = "Audio:" in stderr

        # Extract duration
        for line in stderr.splitlines():
            line = line.strip()
            if "Duration:" in line:
                try:
                    dur_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = dur_str.split(":")
                    duration = float(h) * 3600 + float(m) * 60 + float(s)
                except Exception:
                    pass
            if "Video:" in line and "Stream #" in line:
                # e.g. 1920x1080, 29.97 fps
                try:
                    for part in line.split(","):
                        part = part.strip()
                        if "x" in part:
                            dims = part.split()[0].split("x")
                            if len(dims) == 2 and dims[0].isdigit() and dims[1].isdigit():
                                width, height = int(dims[0]), int(dims[1])
                        if "fps" in part:
                            fps = float(part.replace("fps", "").strip())
                except Exception:
                    pass

        total_frames = int(duration * fps) if duration > 0 and fps > 0 else 0
        if total_frames <= 0:
            total_frames = 100  # Fallback estimate

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,
            total_frames=total_frames,
            duration=duration,
            has_audio=has_audio
        )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        encoder: str = "h264_qsv",
        crf: int = 17,
        enable_face_restore: bool = False,
        face_fidelity: float = 0.8,
        denoise_strength: float = 0.0,
        enhance_lighting: bool = False,
        lighting_strength: float = 0.5,
        replace_color_bgr: Optional[Tuple[int, int, int]] = None,
        remove_object: bool = False,
        swap_face_img: Optional[np.ndarray] = None,
        profile: str = "balanced",
        colorize: bool = False,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> bool:
        """
        Executes end-to-end video upscaling and enhancement:
        1. Demuxes audio to a temporary file.
        2. Spawns FFmpeg decoder pipe outputting rawvideo bgr24 to stdout.
        3. Spawns FFmpeg encoder pipe accepting rawvideo bgr24 from stdin.
        4. Processes each frame through the inference engine with adaptive tiling.
        5. Multiplexes enhanced video and original audio with accurate sync.
        """
        meta = self.probe_video(input_path)
        scale = self.engine.current_scale
        out_w = meta.width * scale
        out_h = meta.height * scale

        temp_audio = output_path + ".audio.m4a"
        temp_video = output_path + ".temp.mp4"

        # 1. Demux audio without re-encoding
        if meta.has_audio:
            audio_cmd = [
                self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-i", input_path,
                "-vn", "-c:a", "copy",
                temp_audio
            ]
            subprocess.run(audio_cmd, capture_output=True)

        # 2. Setup FFmpeg decoder pipe
        decode_cmd = [
            self.ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        dec_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, bufsize=meta.width * meta.height * 3 * 2)

        # 3. Setup FFmpeg encoder pipe
        enc_cmd = [
            self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(meta.fps),
            "-i", "-",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p"
        ]
        # Encoder specific rate-control flags
        if "qsv" in encoder:
            enc_cmd.extend(["-global_quality", str(crf)])
        elif "nvenc" in encoder:
            enc_cmd.extend(["-cq", str(crf), "-preset", "p6"])
        elif "mf" in encoder:
            enc_cmd.extend(["-b:v", "15M"])
        else:
            enc_cmd.extend(["-crf", str(crf), "-preset", "medium"])

        enc_cmd.append(temp_video)
        enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=out_w * out_h * 3 * 2)

        frame_bytes = meta.width * meta.height * 3
        frame_idx = 0
        start_time = time.time()

        try:
            while True:
                if cancel_check and cancel_check():
                    dec_proc.kill()
                    enc_proc.kill()
                    self._cleanup_temp([temp_audio, temp_video])
                    return False

                raw_frame = dec_proc.stdout.read(frame_bytes)
                if not raw_frame or len(raw_frame) < frame_bytes:
                    break

                # Convert raw bytes to BGR numpy array
                frame_np = np.frombuffer(raw_frame, dtype=np.uint8).reshape((meta.height, meta.width, 3))

                # Process single frame through inference engine
                enhanced_frame = self.engine.process_image(
                    img_bgr=frame_np,
                    enable_face_restore=enable_face_restore,
                    face_fidelity=face_fidelity,
                    denoise_strength=denoise_strength,
                    enhance_lighting=enhance_lighting,
                    lighting_strength=lighting_strength,
                    replace_color_bgr=replace_color_bgr,
                    remove_object=remove_object,
                    swap_face_img=swap_face_img,
                    profile=profile,
                    colorize=colorize,
                    progress_callback=None,
                    cancel_check=cancel_check
                )

                # Write processed frame directly to encoder pipe
                enc_proc.stdin.write(enhanced_frame.tobytes())
                frame_idx += 1

                elapsed = max(time.time() - start_time, 0.001)
                fps_actual = frame_idx / elapsed

                if progress_callback:
                    progress_callback(frame_idx, meta.total_frames, fps_actual, elapsed)

        finally:
            if dec_proc.stdout:
                dec_proc.stdout.close()
            dec_proc.wait()

            if enc_proc.stdin:
                enc_proc.stdin.close()
            enc_proc.wait()

        # 4. Remux audio and video tracks
        if os.path.exists(temp_video):
            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 100:
                mux_cmd = [
                    self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", temp_video,
                    "-i", temp_audio,
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-shortest",
                    output_path
                ]
            else:
                mux_cmd = [
                    self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", temp_video,
                    "-c:v", "copy",
                    output_path
                ]
            subprocess.run(mux_cmd, capture_output=True)
            self._cleanup_temp([temp_audio, temp_video])
            return True

        self._cleanup_temp([temp_audio, temp_video])
        return False

    def _cleanup_temp(self, paths: list):
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
