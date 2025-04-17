import os
import logging
import platform
import re
from typing import Dict, Any, Union, cast

logger = logging.getLogger(__name__)


if platform.system() == "Windows":
    import ctypes
    from ctypes import c_ulong, c_ulonglong

    has_windll = hasattr(ctypes, "windll")
else:
    has_windll = False

    c_ulong = Any  # type: ignore
    c_ulonglong = Any  # type: ignore


class ModelVerifier:
    """
    Verifies LLM models for compatibility and quality.

    This class analyzes model files to detect potential issues, checks system
    compatibility, and provides recommendations for optimal performance.
    """

    MINIMUM_MODEL_SIZE_MB: float = 10.0
    RECOMMENDED_RAM_GB: float = 8.0
    CPU_THREAD_TARGET: int = 4

    COMPATIBLE_MODELS: Dict[str, str] = {
        r"\.gguf$": "GGUF format (recommended)",
        r"\.ggml$": "GGML format (legacy)",
        r"\.bin$": "Binary format",
    }

    QUANTIZATION_PATTERNS: Dict[str, Dict[str, Union[float, str]]] = {
        r"q2_k": {
            "gpu_ram_gb": 3.0,
            "cpu_ram_gb": 6.0,
            "quality": "Low",
            "speed": "Fast",
        },
        r"q3_k": {
            "gpu_ram_gb": 4.0,
            "cpu_ram_gb": 7.0,
            "quality": "Medium-Low",
            "speed": "Fast",
        },
        r"q4_k": {
            "gpu_ram_gb": 5.0,
            "cpu_ram_gb": 8.0,
            "quality": "Medium",
            "speed": "Medium",
        },
        r"q5_k": {
            "gpu_ram_gb": 6.0,
            "cpu_ram_gb": 10.0,
            "quality": "Medium-High",
            "speed": "Medium-Slow",
        },
        r"q6_k": {
            "gpu_ram_gb": 7.0,
            "cpu_ram_gb": 12.0,
            "quality": "High",
            "speed": "Slow",
        },
        r"q8_0": {
            "gpu_ram_gb": 9.0,
            "cpu_ram_gb": 16.0,
            "quality": "Very High",
            "speed": "Very Slow",
        },
        r"f16": {
            "gpu_ram_gb": 16.0,
            "cpu_ram_gb": 32.0,
            "quality": "Best",
            "speed": "Slowest",
        },
    }

    def __init__(self) -> None:
        """Initialize the model verifier."""
        self.system_info = self._get_system_info()

    def verify_model(self, model_path: str) -> Dict[str, Any]:
        """
        Verify a model file for compatibility and quality.

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with verification results and recommendations
        """
        result: Dict[str, Any] = {
            "model_path": model_path,
            "exists": False,
            "format_compatible": False,
            "size_ok": False,
            "quantization": "unknown",
            "compatible_with_system": False,
            "warnings": [],
            "recommendations": [],
            "system_info": self.system_info,
        }

        if not os.path.exists(model_path):
            result["warnings"].append(f"Model file not found at: {model_path}")
            result["recommendations"].append(
                "Download or provide the correct path to a GGUF/GGML model file."
            )
            return result

        result["exists"] = True

        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_gb = file_size_mb / 1024

        result["size_bytes"] = file_size_bytes
        result["size_mb"] = file_size_mb
        result["size_gb"] = file_size_gb

        if file_size_mb < self.MINIMUM_MODEL_SIZE_MB:
            result["warnings"].append(
                f"Model file is suspiciously small ({file_size_mb:.2f} MB). "
                "It may be incomplete or corrupted."
            )
            result["recommendations"].append(
                "Download a complete model file. GGUF models are typically 1-8 GB."
            )
        else:
            result["size_ok"] = True

        filename = os.path.basename(model_path).lower()
        format_compatible = False
        format_name = "unknown"

        for pattern, description in self.COMPATIBLE_MODELS.items():
            if re.search(pattern, filename, re.IGNORECASE):
                format_compatible = True
                format_name = description
                break

        result["format_compatible"] = format_compatible
        result["format"] = format_name

        if not format_compatible:
            result["warnings"].append(
                f"Model file format not recognized. Supported formats: "
                f"{', '.join(v for k, v in self.COMPATIBLE_MODELS.items())}"
            )
            result["recommendations"].append(
                "Use a model in GGUF format (recommended) or GGML format."
            )

        quant_level = "unknown"
        ram_requirements: Dict[str, Union[float, str]] = {
            "gpu_ram_gb": 8.0,
            "cpu_ram_gb": 16.0,
        }

        for pattern, info in self.QUANTIZATION_PATTERNS.items():
            if re.search(pattern, filename, re.IGNORECASE):
                quant_level = pattern
                ram_requirements = info
                break

        result["quantization"] = quant_level
        result["ram_requirements"] = ram_requirements

        has_gpu = cast(bool, self.system_info["has_gpu"])
        ram_gb = cast(float, self.system_info["ram_gb"])
        gpu_ram_gb = cast(float, ram_requirements["gpu_ram_gb"])
        cpu_ram_gb = cast(float, ram_requirements["cpu_ram_gb"])
        required_ram = gpu_ram_gb if has_gpu else cpu_ram_gb

        result["compatible_with_system"] = ram_gb >= required_ram

        if not result["compatible_with_system"]:
            result["warnings"].append(
                f"Your system has {ram_gb:.1f} GB RAM, but this model recommends "
                f"at least {required_ram} GB RAM for optimal performance."
            )

            if has_gpu:
                result["recommendations"].append(
                    "Try a more efficient quantized model (Q2_K or Q3_K) to reduce memory usage."
                )
            else:
                result["recommendations"].append(
                    "Try a more efficient quantized model or increase your system RAM."
                )

        if not has_gpu and ram_gb < self.RECOMMENDED_RAM_GB:
            result["warnings"].append(
                f"Running on CPU with {ram_gb:.1f} GB RAM may be slow."
            )
            result["recommendations"].append(
                "Consider using a smaller model or reducing context window size."
            )

        try:
            if format_compatible and re.search(r"\.gguf$", filename, re.IGNORECASE):
                metadata = self._extract_model_metadata(model_path)
                result["metadata"] = metadata

                if metadata:
                    if metadata.get("quantization_version") == "1":
                        result["warnings"].append(
                            "Using an older quantization version. Newer versions may perform better."
                        )

                    context_length_str = metadata.get("context_length", "0")
                    if context_length_str:
                        try:
                            context_length = int(context_length_str)
                            if context_length > 8192:
                                result["warnings"].append(
                                    f"Model has large context window ({context_length}). "
                                    "This may require more memory."
                                )
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"Could not extract metadata: {str(e)}")

        if result["exists"] and result["format_compatible"] and result["size_ok"]:
            if result["compatible_with_system"]:
                logger.info(f"Model verification passed: {model_path}")
            else:
                logger.warning(
                    f"Model may work but with performance issues: {model_path}"
                )
        else:
            logger.error(f"Model verification failed: {model_path}")

        return result

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system.

        Returns:
            Dictionary with system information
        """
        system_info: Dict[str, Any] = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 4,
            "ram_gb": self._get_system_ram_gb(),
            "has_gpu": self._detect_gpu(),
            "has_metal": platform.system() == "Darwin"
            and platform.machine() == "arm64",
        }

        return system_info

    def _get_system_ram_gb(self) -> float:
        """
        Get the system RAM in GB.

        Returns:
            System RAM in GB
        """
        try:
            if platform.system() == "Windows" and has_windll:
                import ctypes
                from ctypes import c_ulong, c_ulonglong

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]

                kernel32 = ctypes.windll.kernel32  # type: ignore
                memoryStatus = MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                return float(memoryStatus.ullTotalPhys) / (1024**3)

            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = float(line.split()[1])
                            return mem_kb / (1024**2)

            elif platform.system() == "Darwin":
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout:
                    return float(result.stdout.strip()) / (1024**3)

        except Exception as e:
            logger.debug(f"Error getting system RAM: {str(e)}")

        return 8.0

    def _detect_gpu(self) -> bool:
        """
        Detect if a compatible GPU is available.

        Returns:
            True if GPU is available, False otherwise
        """
        try:
            if platform.system() in ["Linux", "Windows"]:
                try:
                    import subprocess

                    result = subprocess.run(
                        ["nvidia-smi"], capture_output=True, timeout=2, check=False
                    )
                    return result.returncode == 0
                except (FileNotFoundError, subprocess.SubprocessError):
                    pass

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return True

            if platform.system() == "Linux":
                try:
                    return os.path.exists("/dev/dri/renderD128")
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting GPU: {str(e)}")

        return False

    def _extract_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a GGUF model file.
        This is a basic check and might not work for all models.

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with model metadata
        """
        metadata: Dict[str, Any] = {}

        try:
            with open(model_path, "rb") as f:
                header = f.read(8192)

                if header[:4] == b"GGUF":
                    metadata["format_detected"] = "GGUF"

                    version_match = re.search(rb"version.{1,6}(\d+)", header)
                    if version_match:
                        metadata["version"] = version_match.group(1).decode(
                            "utf-8", errors="ignore"
                        )

                    quant_match = re.search(rb"quant.{1,12}(\d+)", header)
                    if quant_match:
                        metadata["quantization_version"] = quant_match.group(1).decode(
                            "utf-8", errors="ignore"
                        )

                    ctx_match = re.search(rb"context.{1,12}(\d+)", header)
                    if ctx_match:
                        metadata["context_length"] = ctx_match.group(1).decode(
                            "utf-8", errors="ignore"
                        )

                elif header[:4] == b"GGML":
                    metadata["format_detected"] = "GGML"

        except Exception as e:
            logger.debug(f"Error extracting metadata: {str(e)}")

        return metadata
