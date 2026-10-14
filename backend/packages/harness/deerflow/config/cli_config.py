from typing import Literal

from pydantic import BaseModel, Field


class CliConfig(BaseModel):
    """Startup configuration for the external CLI-session harness.

    The gateway launches a headless coding CLI (``muse exec --json``) per
    session row, either inside a named tmux session so runs survive a
    gateway restart, or as a direct background subprocess.
    """

    tmux: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="tmux usage for CLI runs: 'auto' uses tmux when the binary is present else direct subprocess; 'on' fails the launch when tmux is missing; 'off' always uses a direct subprocess.",
    )
    binary: str = Field(
        default="muse",
        description="CLI executable name or absolute path used to build launch commands.",
    )
    probe_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Wall-clock timeout for short control probes (tmux has-session, trace inspect, export).",
    )
    events_poll_seconds: float = Field(
        default=0.25,
        ge=0.05,
        le=5.0,
        description="Poll interval for tailing a run log file on the SSE events endpoint.",
    )
