import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from uv_pack._logging import console_print


def _diagnostics_enabled() -> bool:
    return os.environ.get("UV_PACK_DIAGNOSTICS") == "1"


_DIAGNOSTICS_START = time.monotonic()


def _diagnostic_log(message: str) -> None:
    if not _diagnostics_enabled():
        return
    elapsed = time.monotonic() - _DIAGNOSTICS_START
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    print(
        f"[uv-pack diagnostic {timestamp} +{elapsed:.3f}s] {message}",
        file=sys.stderr,
        flush=True,
    )


def _log_top_snapshot() -> None:
    try:
        result = subprocess.run(
            ["top", "-b", "-n", "1", "-w", "180"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        _diagnostic_log(f"top snapshot unavailable: {exc}")
        return

    lines = result.stdout.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("PID ")),
        None,
    )
    if header_index is None:
        _diagnostic_log(f"top snapshot failed (exit {result.returncode}): {result.stderr.strip()}")
        return

    snapshot = "\n".join(lines[: min(header_index + 9, len(lines))])
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    elapsed = time.monotonic() - _DIAGNOSTICS_START
    print(
        f"[uv-pack diagnostic {timestamp} +{elapsed:.3f}s] top CPU snapshot:\n{snapshot}",
        file=sys.stderr,
        flush=True,
    )


def _sample_top_until_stopped(stop: threading.Event) -> None:
    while not stop.wait(10):
        _log_top_snapshot()


@dataclass(frozen=True)
class CommandOutput:
    """Captured subprocess execution result."""

    name: str
    stdout: str
    stderr: str
    returncode: int


@contextmanager
def run_step(cmd_name: str, *, should_run: bool = True) -> Iterator[Progress | None]:
    """Context manager to show a progress display during step execution."""
    diagnostics = _diagnostics_enabled()
    ctx = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            disable=diagnostics,
        )
        if should_run
        else nullcontext()
    )

    _diagnostic_log(f"run_step {cmd_name}: before context enter")
    try:
        with ctx as progress:
            _diagnostic_log(f"run_step {cmd_name}: context entered")
            if progress is not None:
                progress.add_task(f"Running '{cmd_name}'...", total=None)
                _diagnostic_log(f"run_step {cmd_name}: progress task added")
            _diagnostic_log(f"run_step {cmd_name}: before yielding to step body")
            try:
                yield progress
            finally:
                _diagnostic_log(f"run_step {cmd_name}: step body returned")
    finally:
        _diagnostic_log(f"run_step {cmd_name}: context exited")


def run_cmd(
    cmd: list[str],
    cmd_name: str,
    *,
    stream_output: bool = False,
) -> CommandOutput:
    """Execute a subprocess command, optionally streaming its output."""
    diagnostics = _diagnostics_enabled()
    _diagnostic_log(f"command {cmd_name}: started")
    monitor_stop = threading.Event()
    monitor = None
    if diagnostics:
        monitor = threading.Thread(
            target=_sample_top_until_stopped,
            args=(monitor_stop,),
            daemon=True,
        )
        monitor.start()

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=not (stream_output or diagnostics),
            check=False,
        )
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join()

    _diagnostic_log(
        f"command {cmd_name}: finished in {time.monotonic() - started:.3f}s "
        f"with exit code {proc.returncode}",
    )
    return CommandOutput(
        name=cmd_name,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )


def exit_on_error(result: CommandOutput) -> None:
    """Exit the CLI if a subprocess failed."""
    if result.returncode == 0:
        return

    if result.stderr:
        if sys.stderr.isatty():
            console_print(
                Panel.fit(
                    result.stderr.rstrip(),
                    title=f"[bold red]✘ '{result.name}' failed[/bold red]",
                    border_style="red",
                ),
            )
        else:
            typer.echo(result.stderr, err=True)

    raise typer.Exit(code=result.returncode)
