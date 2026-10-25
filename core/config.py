import os
import sys
import shutil
import platform
import subprocess
from typing import Dict, Any, List, Optional

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

class SystemDiagnostics:
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        info = {
            "name": platform.processor() or "x86_64 Processor",
            "physical_cores": os.cpu_count() or 4,
            "logical_cores": os.cpu_count() or 4,
            "platform": platform.platform()
        }
        if sys.platform == "win32":
            try:
                cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
                if out:
                    info["name"] = out.splitlines()[0].strip()
            except Exception:
                pass
        return info

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        gpus = []
        if sys.platform == "win32":
            try:
                cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
                for line in out.splitlines():
                    clean = line.strip()
                    if clean:
                        vendor = "Intel" if "intel" in clean.lower() else ("Nvidia" if "nvidia" in clean.lower() else ("AMD" if "amd" in clean.lower() else "Generic"))
                        is_virtual = "virtual" in clean.lower() or "rdp" in clean.lower() or "vbox" in clean.lower()
                        gpus.append({"name": clean, "vendor": vendor, "is_virtual": is_virtual})
                # Sort physical GPUs to top
                gpus.sort(key=lambda x: (x["is_virtual"], x["vendor"] == "Generic"))
            except Exception:
                pass
        if not gpus:
            gpus.append({"name": "Integrated Display Device", "vendor": "Generic", "is_virtual": False})
        return gpus

    @staticmethod
    def get_onnx_providers() -> List[str]:
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

    @staticmethod
    def resolve_ffmpeg_path() -> Optional[str]:
        # 1. System PATH
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg and os.path.exists(system_ffmpeg):
            return system_ffmpeg
        
        # 2. Bundled imageio_ffmpeg
        if imageio_ffmpeg is not None:
            try:
                bundled = imageio_ffmpeg.get_ffmpeg_exe()
                if bundled and os.path.exists(bundled):
                    return bundled
            except Exception:
                pass

        # 3. Known Windows directories
        local_bins = [
            os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
            os.path.join(os.path.dirname(sys.executable), "Scripts", "ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.getcwd(), "bin", "ffmpeg.exe"),
        ]
        for p in local_bins:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def detect_hardware_encoders(ffmpeg_bin: Optional[str]) -> List[str]:
        if not ffmpeg_bin or not os.path.exists(ffmpeg_bin):
            return ["libx264"]
        
        available = []
        try:
            res = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
            stdout = res.stdout.lower()
            if "h264_qsv" in stdout:
                available.append("h264_qsv (Intel QuickSync)")
            if "hevc_qsv" in stdout:
                available.append("hevc_qsv (Intel QuickSync HEVC)")
            if "h264_nvenc" in stdout:
                available.append("h264_nvenc (NVIDIA NVENC)")
            if "h264_amf" in stdout:
                available.append("h264_amf (AMD AMF)")
            if "h264_mf" in stdout:
                available.append("h264_mf (Windows MediaFoundation)")
            if "libx264" in stdout:
                available.append("libx264 (Software CPU)")
        except Exception:
            available.append("libx264 (Software CPU)")
        
        return available if available else ["libx264 (Software CPU)"]


class AppConfig:
    def __init__(self):
        self.cpu_info = SystemDiagnostics.get_cpu_info()
        self.gpus = SystemDiagnostics.get_gpu_info()
        self.onnx_providers = SystemDiagnostics.get_onnx_providers()
        self.ffmpeg_path = SystemDiagnostics.resolve_ffmpeg_path()
        self.available_encoders = SystemDiagnostics.detect_hardware_encoders(self.ffmpeg_path)

        # Hardware-aware presets tailored for Intel Iris Xe / 16GB RAM laptop
        self.is_iris_xe = any("iris" in g["name"].lower() or "intel" in g["name"].lower() for g in self.gpus)
        self.has_directml = "DmlExecutionProvider" in self.onnx_providers
        self.has_cuda = "CUDAExecutionProvider" in self.onnx_providers

        # Safe default tile size to prevent TDR graphics timeout and OOM
        if self.has_directml:
            self.default_tile_size = 256
            self.default_tile_overlap = 32
            self.preferred_provider = "DmlExecutionProvider"
        elif self.has_cuda:
            self.default_tile_size = 512
            self.default_tile_overlap = 32
            self.preferred_provider = "CUDAExecutionProvider"
        else:
            self.default_tile_size = 256
            self.default_tile_overlap = 24
            self.preferred_provider = "CPUExecutionProvider"

        # Safe video batch size
        self.default_batch_size = 1  # 1 frame at a time on Iris Xe to guarantee stability
        if getattr(sys, "frozen", False):
            exe_models = os.path.join(os.path.dirname(sys.executable), "models")
            internal_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
            self.models_dir = exe_models if os.path.exists(exe_models) else internal_models
        else:
            self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        os.makedirs(self.models_dir, exist_ok=True)

    def get_recommended_encoder(self) -> str:
        for enc in self.available_encoders:
            if "qsv" in enc and self.is_iris_xe:
                return enc.split()[0]
            if "nvenc" in enc and self.has_cuda:
                return enc.split()[0]
        for enc in self.available_encoders:
            if "mf" in enc:
                return enc.split()[0]
        return "libx264"
