import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

NVIDIA_SMI_TIMEOUT_S = 5.0
CPUINFO = "/proc/cpuinfo"


@dataclass(frozen=True, slots=True)
class GpuReading:
    name: str
    total_mib: float
    used_mib: float


class HostInspector(Protocol):
    def cpu_name(self) -> str: ...

    def cores(self) -> int: ...

    def gpu(self) -> GpuReading | None: ...


class SystemHost:
    """Reads the CPU model from /proc and GPU memory from nvidia-smi."""

    def cpu_name(self) -> str:
        try:
            with Path(CPUINFO).open(encoding="utf-8") as handle:
                for row in handle:
                    if row.startswith("model name"):
                        return row.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or platform.machine()

    def cores(self) -> int:
        return os.cpu_count() or 1

    def gpu(self) -> GpuReading | None:
        binary = shutil.which("nvidia-smi")
        if binary is None:
            return None
        query = "--query-gpu=name,memory.total,memory.used"
        try:
            done = subprocess.run(
                [binary, query, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_S,
                check=False,
            )
        except OSError, subprocess.TimeoutExpired:
            return None
        first = done.stdout.strip().splitlines()[:1]
        if done.returncode != 0 or not first:
            return None
        return parse_gpu_row(first[0])


def parse_gpu_row(row: str) -> GpuReading | None:
    """Parse one `name, total, used` csv row from nvidia-smi."""
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 3:
        return None
    try:
        return GpuReading(parts[0], float(parts[1]), float(parts[2]))
    except ValueError:
        return None
