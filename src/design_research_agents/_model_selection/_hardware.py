"""Hardware profiling helpers for model selection."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol, cast


@dataclass(slots=True, frozen=True)
class HardwareProfile:
    """Snapshot of system hardware capacity for model selection.

    Attributes:
        total_ram_gb: Total system RAM in GiB.
        available_ram_gb: Available system RAM in GiB.
        cpu_count: Logical CPU count.
        load_average: Load average tuple when supported.
        gpu_present: Whether a GPU is detected.
        gpu_vram_gb: Detected GPU VRAM in GiB.
        gpu_name: Optional GPU name.
        platform_name: Platform identifier string.
    """

    total_ram_gb: float | None
    """Field value for ``total_ram_gb``."""
    available_ram_gb: float | None
    """Field value for ``available_ram_gb``."""
    cpu_count: int | None
    """Field value for ``cpu_count``."""
    load_average: tuple[float, float, float] | None
    """Field value for ``load_average``."""
    gpu_present: bool | None
    """Field value for ``gpu_present``."""
    gpu_vram_gb: float | None
    """Field value for ``gpu_vram_gb``."""
    gpu_name: str | None = None
    """Field value for ``gpu_name``."""
    platform_name: str | None = None
    """Field value for ``platform_name``."""

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Collect a best-effort hardware profile for the current system.

        Returns:
            Detected hardware profile snapshot.
        """
        # Collect memory numbers first so we can budget local models later.
        total_ram_bytes = _detect_total_ram_bytes()
        available_ram_bytes = _detect_available_ram_bytes()
        # GPU detection is best-effort and may return unknowns.
        gpu_present, gpu_vram_bytes, gpu_name = _detect_gpu_info()
        return cls(
            total_ram_gb=_bytes_to_gib(total_ram_bytes),
            available_ram_gb=_bytes_to_gib(available_ram_bytes),
            cpu_count=os.cpu_count(),
            load_average=_detect_load_average(),
            gpu_present=gpu_present,
            gpu_vram_gb=_bytes_to_gib(gpu_vram_bytes),
            gpu_name=gpu_name,
            platform_name=platform.system(),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the profile.

        Returns:
            JSON-serializable hardware profile mapping.
        """
        return {
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "cpu_count": self.cpu_count,
            "load_average": self.load_average,
            "gpu_present": self.gpu_present,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_name": self.gpu_name,
            "platform_name": self.platform_name,
        }

    def __str__(self) -> str:
        """Return a readable JSON representation of the profile.

        Returns:
            Pretty-printed JSON string for the profile.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)


class _WindowsMemoryStatus(Protocol):
    """_WindowsMemoryStatus class."""

    ullTotalPhys: int
    ullAvailPhys: int


def _bytes_to_gib(value: int | None) -> float | None:
    """Run bytes to gib.

    Args:
        value: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if value is None:
        return None
    return value / (1024**3)


def _detect_load_average() -> tuple[float, float, float] | None:
    """Run detect load average.

    Returns:
        Computed return value.
    """
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _detect_total_ram_bytes() -> int | None:
    """Run detect total ram bytes.

    Returns:
        Computed return value.
    """
    system = platform.system()
    # Prefer OS-specific APIs for more accurate totals.
    if system == "Windows":
        return _detect_windows_total_ram_bytes()
    if system == "Darwin":
        return _detect_macos_total_ram_bytes()
    meminfo = _read_proc_meminfo()
    # Linux fast path via /proc is preferred when available.
    if meminfo and "MemTotal" in meminfo:
        return meminfo["MemTotal"] * 1024
    return _detect_sysconf_total_ram_bytes()


def _detect_available_ram_bytes() -> int | None:
    """Run detect available ram bytes.

    Returns:
        Computed return value.
    """
    system = platform.system()
    # Prefer OS-specific APIs for more accurate availability.
    if system == "Windows":
        return _detect_windows_available_ram_bytes()
    if system == "Darwin":
        return _detect_macos_available_ram_bytes()
    meminfo = _read_proc_meminfo()
    # MemAvailable tracks reclaimable memory more accurately than MemFree.
    if meminfo and "MemAvailable" in meminfo:
        return meminfo["MemAvailable"] * 1024
    return _detect_sysconf_available_ram_bytes()


def _read_proc_meminfo() -> dict[str, int] | None:
    """Run read proc meminfo.

    Returns:
        Computed return value.
    """
    if not os.path.exists("/proc/meminfo"):
        return None
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    meminfo[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return None
    return meminfo


def _detect_sysconf_total_ram_bytes() -> int | None:
    """Run detect sysconf total ram bytes.

    Returns:
        Computed return value.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int):
        return pages * page_size
    return None


def _detect_sysconf_available_ram_bytes() -> int | None:
    """Run detect sysconf available ram bytes.

    Returns:
        Computed return value.
    """
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int):
        return pages * page_size
    return None


def _detect_windows_total_ram_bytes() -> int | None:
    """Run detect windows total ram bytes.

    Returns:
        Computed return value.
    """
    status = _windows_memory_status()
    if status is None:
        return None
    return status.ullTotalPhys


def _detect_windows_available_ram_bytes() -> int | None:
    """Run detect windows available ram bytes.

    Returns:
        Computed return value.
    """
    status = _windows_memory_status()
    if status is None:
        return None
    return status.ullAvailPhys


def _detect_macos_total_ram_bytes() -> int | None:
    """Run detect macos total ram bytes.

    Returns:
        Computed return value.
    """
    output = _run_command(["sysctl", "-n", "hw.memsize"])
    if output is None:
        return None
    try:
        return int(output.strip())
    except ValueError:
        return None


def _detect_macos_available_ram_bytes() -> int | None:
    """Run detect macos available ram bytes.

    Returns:
        Computed return value.
    """
    output = _run_command(["vm_stat"])
    if output is None:
        return None
    page_size = 4096
    match = re.search(r"page size of (\d+) bytes", output)
    if match:
        page_size = int(match.group(1))
    pages = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = re.sub(r"[^\d]", "", value)
        if not digits:
            continue
        pages[key.strip()] = int(digits)
    free_pages = pages.get("Pages free", 0)
    inactive_pages = pages.get("Pages inactive", 0)
    speculative_pages = pages.get("Pages speculative", 0)
    available_pages = free_pages + inactive_pages + speculative_pages
    if available_pages <= 0:
        return None
    return available_pages * page_size


def _run_command(args: list[str]) -> str | None:
    """Run run command.

    Args:
        args: Input value for this parameter.

    Returns:
        Computed return value.
    """
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Best-effort probing should fail quietly and let fallback detectors continue.
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _detect_gpu_info() -> tuple[bool | None, int | None, str | None]:
    # Probe NVIDIA first, then fall back to platform-specific inspection.
    """Run detect gpu info.

    Returns:
        Computed return value.
    """
    nvidia_info = _detect_nvidia_gpu_info()
    if nvidia_info is not None:
        return nvidia_info
    system = platform.system()
    if system == "Darwin":
        mac_info = _detect_macos_gpu_info()
        if mac_info is not None:
            return mac_info
    if system == "Windows":
        windows_info = _detect_windows_gpu_info()
        if windows_info is not None:
            return windows_info
    return None, None, None


def _detect_nvidia_gpu_info() -> tuple[bool, int | None, str | None] | None:
    """Run detect nvidia gpu info.

    Returns:
        Computed return value.
    """
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return None
    vram_values: list[int] = []
    names: list[str] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        raw_name, memory = parts[0], parts[1]
        try:
            vram_values.append(int(float(memory)) * 1024 * 1024)
        except ValueError:
            # Keep row alignment even when one device reports malformed memory fields.
            vram_values.append(0)
        if raw_name:
            names.append(raw_name)
    if not vram_values and not names:
        return None
    # Use max VRAM across visible devices to reflect best available local execution target.
    vram = max(vram_values) if vram_values else None
    resolved_name: str | None = names[0] if len(set(names)) == 1 else (names[0] if names else None)
    return True, vram, resolved_name


def _detect_macos_gpu_info() -> tuple[bool, int | None, str | None] | None:
    """Run detect macos gpu info.

    Returns:
        Computed return value.
    """
    output = _run_command(["system_profiler", "SPDisplaysDataType"])
    if output is None:
        return None
    name = None
    vram_bytes = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            name = stripped.split(":", 1)[1].strip()
        if "VRAM" in stripped:
            match = re.search(r"VRAM.*?:\s*([\d\.]+)\s*(GB|MB)", stripped)
            if match:
                size = float(match.group(1))
                unit = match.group(2)
                vram_bytes = int(size * 1024**3) if unit == "GB" else int(size * 1024**2)
    if name is None and vram_bytes is None:
        return None
    return True, vram_bytes, name


def _detect_windows_gpu_info() -> tuple[bool, int | None, str | None] | None:
    """Run detect windows gpu info.

    Returns:
        Computed return value.
    """
    output = _run_command(
        [
            "wmic",
            "path",
            "win32_VideoController",
            "get",
            "Name,AdapterRAM",
            "/format:csv",
        ]
    )
    if output is None:
        return None
    vram_values: list[int] = []
    names: list[str] = []
    for line in output.splitlines():
        if not line.strip() or line.startswith("Node"):
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 3:
            continue
        adapter_ram = parts[1]
        raw_name = parts[2]
        try:
            vram_values.append(int(adapter_ram))
        except ValueError:
            vram_values.append(0)
        if raw_name:
            names.append(raw_name)
    if not vram_values and not names:
        return None
    vram = max(vram_values) if vram_values else None
    resolved_name: str | None = names[0] if len(set(names)) == 1 else (names[0] if names else None)
    return True, vram, resolved_name


def _windows_memory_status() -> _WindowsMemoryStatus | None:
    """Run windows memory status.

    Returns:
        Computed return value.
    """
    if platform.system() != "Windows":
        return None
    try:
        import ctypes
    except ImportError:
        return None

    class _MemoryStatus(ctypes.Structure):
        """_MemoryStatus class."""

        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    kernel32 = getattr(windll, "kernel32", None)
    if kernel32 is None:
        return None
    global_status = getattr(kernel32, "GlobalMemoryStatusEx", None)
    if global_status is None or global_status(ctypes.byref(status)) == 0:
        return None
    # Cast into protocol shape to avoid exposing ctypes-specific type details downstream.
    return cast(_WindowsMemoryStatus, status)
