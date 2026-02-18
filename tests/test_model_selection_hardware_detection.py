from __future__ import annotations

import builtins
import io
import subprocess
from types import SimpleNamespace

import pytest

from design_research_agents.model_selection import hardware as hw


def test_bytes_to_gib_and_load_average_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hw._bytes_to_gib(None) is None
    assert hw._bytes_to_gib(1024**3) == 1.0

    monkeypatch.setattr(hw.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("no loadavg")))
    assert hw._detect_load_average() is None


def test_read_proc_meminfo_absent_or_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw.os.path, "exists", lambda _: False)
    assert hw._read_proc_meminfo() is None

    monkeypatch.setattr(hw.os.path, "exists", lambda _: True)

    def _raise_open(*args: object, **kwargs: object) -> io.StringIO:
        del args, kwargs
        raise OSError("boom")

    monkeypatch.setattr(builtins, "open", _raise_open)
    assert hw._read_proc_meminfo() is None


def test_read_proc_meminfo_parses_expected_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "\n".join(
        [
            "MemTotal:       16384 kB",
            "MemAvailable:   8192 kB",
            "MalformedLine",
            "BadNumber: not-an-int",
            "",
        ]
    )
    monkeypatch.setattr(hw.os.path, "exists", lambda _: True)
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.StringIO(content))

    result = hw._read_proc_meminfo()

    assert result == {"MemTotal": 16384, "MemAvailable": 8192}


def test_sysconf_ram_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "SC_PHYS_PAGES": 100,
        "SC_AVPHYS_PAGES": 25,
        "SC_PAGE_SIZE": 4096,
    }
    monkeypatch.setattr(hw.os, "sysconf", lambda key: values[key])

    assert hw._detect_sysconf_total_ram_bytes() == 409600
    assert hw._detect_sysconf_available_ram_bytes() == 102400

    monkeypatch.setattr(hw.os, "sysconf", lambda _key: "not-int")
    assert hw._detect_sysconf_total_ram_bytes() is None

    monkeypatch.setattr(
        hw.os,
        "sysconf",
        lambda _key: (_ for _ in ()).throw(OSError("sysconf failed")),
    )
    assert hw._detect_sysconf_available_ram_bytes() is None


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Windows", 111),
        ("Darwin", 222),
        ("Linux", 333 * 1024),
    ],
)
def test_detect_total_ram_bytes_prefers_platform_probes(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    expected: int,
) -> None:
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(hw, "_detect_windows_total_ram_bytes", lambda: 111)
    monkeypatch.setattr(hw, "_detect_macos_total_ram_bytes", lambda: 222)
    monkeypatch.setattr(hw, "_detect_sysconf_total_ram_bytes", lambda: 444)
    monkeypatch.setattr(
        hw,
        "_read_proc_meminfo",
        lambda: {"MemTotal": 333} if system == "Linux" else None,
    )

    assert hw._detect_total_ram_bytes() == expected


def test_detect_total_ram_bytes_linux_falls_back_to_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_read_proc_meminfo", lambda: {})
    monkeypatch.setattr(hw, "_detect_sysconf_total_ram_bytes", lambda: 9876)

    assert hw._detect_total_ram_bytes() == 9876


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Windows", 101),
        ("Darwin", 202),
        ("Linux", 303 * 1024),
    ],
)
def test_detect_available_ram_bytes_prefers_platform_probes(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    expected: int,
) -> None:
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(hw, "_detect_windows_available_ram_bytes", lambda: 101)
    monkeypatch.setattr(hw, "_detect_macos_available_ram_bytes", lambda: 202)
    monkeypatch.setattr(hw, "_detect_sysconf_available_ram_bytes", lambda: 404)
    monkeypatch.setattr(
        hw,
        "_read_proc_meminfo",
        lambda: {"MemAvailable": 303} if system == "Linux" else None,
    )

    assert hw._detect_available_ram_bytes() == expected


def test_detect_available_ram_bytes_linux_falls_back_to_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_read_proc_meminfo", lambda: {"MemTotal": 999})
    monkeypatch.setattr(hw, "_detect_sysconf_available_ram_bytes", lambda: 8765)

    assert hw._detect_available_ram_bytes() == 8765


def test_run_command_success_failure_and_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    assert hw._run_command(["echo", "ok"]) == "ok"

    monkeypatch.setattr(
        hw.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )
    assert hw._run_command(["false"]) is None

    monkeypatch.setattr(
        hw.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="x", timeout=1)
        ),
    )
    assert hw._run_command(["sleep", "9"]) is None


def test_detect_macos_total_ram_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_run_command", lambda _args: "17179869184\n")
    assert hw._detect_macos_total_ram_bytes() == 17179869184

    monkeypatch.setattr(hw, "_run_command", lambda _args: "not-a-number")
    assert hw._detect_macos_total_ram_bytes() is None

    monkeypatch.setattr(hw, "_run_command", lambda _args: None)
    assert hw._detect_macos_total_ram_bytes() is None


def test_detect_macos_available_ram_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    vm_stat = "\n".join(
        [
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
            "Pages free: 3.",
            "Pages inactive: 2.",
            "Pages speculative: 1.",
        ]
    )
    monkeypatch.setattr(hw, "_run_command", lambda _args: vm_stat)
    assert hw._detect_macos_available_ram_bytes() == 6 * 16384

    no_available = "\n".join(
        [
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)",
            "Pages free: 0.",
            "Pages inactive: 0.",
            "Pages speculative: 0.",
        ]
    )
    monkeypatch.setattr(hw, "_run_command", lambda _args: no_available)
    assert hw._detect_macos_available_ram_bytes() is None


def test_detect_nvidia_gpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_run_command", lambda _args: "RTX 4090, 24576\nRTX 4090, 20480")
    assert hw._detect_nvidia_gpu_info() == (True, 24576 * 1024 * 1024, "RTX 4090")

    monkeypatch.setattr(hw, "_run_command", lambda _args: "A, bad\nB, 10")
    detected = hw._detect_nvidia_gpu_info()
    assert detected == (True, 10 * 1024 * 1024, "A")

    monkeypatch.setattr(hw, "_run_command", lambda _args: "")
    assert hw._detect_nvidia_gpu_info() is None


def test_detect_macos_gpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n".join(
        [
            "Chipset Model: Apple M3",
            "VRAM (Dynamic, Max): 8 GB",
        ]
    )
    monkeypatch.setattr(hw, "_run_command", lambda _args: output)
    assert hw._detect_macos_gpu_info() == (True, 8 * 1024**3, "Apple M3")

    output_mb = "VRAM (Total): 512 MB"
    monkeypatch.setattr(hw, "_run_command", lambda _args: output_mb)
    assert hw._detect_macos_gpu_info() == (True, 512 * 1024**2, None)

    monkeypatch.setattr(hw, "_run_command", lambda _args: None)
    assert hw._detect_macos_gpu_info() is None


def test_detect_windows_gpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_output = "\n".join(
        [
            "Node,AdapterRAM,Name",
            "HOST,4293918720,NVIDIA GeForce RTX",
            "HOST,2147483648,Intel Arc",
        ]
    )
    monkeypatch.setattr(hw, "_run_command", lambda _args: csv_output)
    assert hw._detect_windows_gpu_info() == (True, 4293918720, "NVIDIA GeForce RTX")

    monkeypatch.setattr(hw, "_run_command", lambda _args: "Node,AdapterRAM,Name")
    assert hw._detect_windows_gpu_info() is None


def test_detect_gpu_info_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_detect_nvidia_gpu_info", lambda: (True, 1, "nvidia"))
    assert hw._detect_gpu_info() == (True, 1, "nvidia")

    monkeypatch.setattr(hw, "_detect_nvidia_gpu_info", lambda: None)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw, "_detect_macos_gpu_info", lambda: (True, 2, "apple"))
    assert hw._detect_gpu_info() == (True, 2, "apple")

    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_detect_windows_gpu_info", lambda: (True, 3, "win"))
    assert hw._detect_gpu_info() == (True, 3, "win")

    monkeypatch.setattr(hw, "_detect_windows_gpu_info", lambda: None)
    assert hw._detect_gpu_info() == (None, None, None)


def test_windows_memory_status_non_windows_and_windows_without_windll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._windows_memory_status() is None

    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    assert hw._windows_memory_status() is None


def test_hardware_profile_detect_aggregates_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw, "_detect_total_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(hw, "_detect_available_ram_bytes", lambda: 6 * 1024**3)
    monkeypatch.setattr(hw, "_detect_gpu_info", lambda: (True, 8 * 1024**3, "GPU-X"))
    monkeypatch.setattr(hw, "_detect_load_average", lambda: (0.1, 0.2, 0.3))
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(hw.platform, "system", lambda: "UnitTestOS")

    profile = hw.HardwareProfile.detect()

    assert profile.total_ram_gb == 16.0
    assert profile.available_ram_gb == 6.0
    assert profile.gpu_present is True
    assert profile.gpu_vram_gb == 8.0
    assert profile.gpu_name == "GPU-X"
    assert profile.cpu_count == 12
    assert profile.platform_name == "UnitTestOS"
