#!/usr/bin/env bash
# quantflow-app.sh — QuantFlow desktop entry point.
#
# Reuses a healthy stack when one is already running (e.g. the
# deer-flow-dev.service user service), otherwise starts the stack via
# scripts/serve.sh in daemon mode, then opens the UI in a dedicated
# Chromium app window. Only one owner instance manages teardown, and only
# for a stack THIS script started — a reused pre-existing stack is NEVER
# stopped on window close or by --stop.
#
# Usage: quantflow-app.sh [--dev|--prod] [--no-browser] [--stop]
#                         [--skip-frontend-build] [--help]
#   --dev                 force dev mode (hot-reload gateway + next dev)
#   --prod                force prod mode (pre-built frontend, no hot-reload)
#   --no-browser          ensure the stack is up, then exit (no window)
#   --stop                stop the stack ONLY if this script started it
#   --skip-frontend-build with --prod, reuse the existing .next build
#                         (default when a build exists; fails otherwise)
#   -h, --help            print this usage and exit
#
# Env: QUANTFLOW_BROWSER_CMD  override browser command (used by headless tests)
#
# Ports: gateway 8001, frontend 3000, nginx unified 2026.
set -uo pipefail

REPO_ROOT="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
cd "$REPO_ROOT" || exit 1

GATEWAY_PORT=8001
FRONTEND_PORT=3000
NGINX_PORT=2026
URL="http://localhost:${NGINX_PORT}"
PROFILE="$HOME/.config/quantflow-chrome"
LOCK_DIR=/tmp/opencode
LOCK=$LOCK_DIR/quantflow.lock
MARKER=$LOCK_DIR/quantflow-app.started
WAIT_FOR_PORT="$REPO_ROOT/scripts/wait-for-port.sh"
SERVE_SH="$REPO_ROOT/scripts/serve.sh"

MODE=""               # dev | prod | "" (auto)
NO_BROWSER=0
DO_STOP=0
SKIP_FRONTEND_BUILD=""  # "" (auto) | "1" (user passed --skip-frontend-build)
STARTED_BY_ME=0

usage() {
    sed -n '2,20p' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dev) MODE="dev"; shift ;;
        --prod) MODE="prod"; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        --stop) DO_STOP=1; shift ;;
        --skip-frontend-build) SKIP_FRONTEND_BUILD="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

mkdir -p "$LOCK_DIR"

# ── Port helpers ─────────────────────────────────────────────────────────────

_is_listening() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 && return 0
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q . && return 0
    fi
    if command -v netstat >/dev/null 2>&1; then
        if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|[.:])${port}$"; then
            return 0
        fi
    fi
    return 1
}

# True when ANY stack port already answers — i.e. a stack (ours or someone
# else's, e.g. deer-flow-dev.service) owns these ports and must be reused,
# never displaced.
_stack_live() {
    _is_listening "$NGINX_PORT" || _is_listening "$GATEWAY_PORT" || _is_listening "$FRONTEND_PORT"
}

_wait_healthy() {
    # Bounded waits via the repo helper. Prints log paths on timeout.
    bash "$WAIT_FOR_PORT" "$GATEWAY_PORT" "$1" "Gateway" || return 1
    bash "$WAIT_FOR_PORT" "$FRONTEND_PORT" "$2" "Frontend" || return 1
    bash "$WAIT_FOR_PORT" "$NGINX_PORT" "$3" "Nginx" || return 1
    return 0
}

_print_log_paths() {
    echo "  Logs (if this checkout produced them):" >&2
    echo "    $REPO_ROOT/logs/gateway.log" >&2
    echo "    $REPO_ROOT/logs/frontend.log" >&2
    echo "    $REPO_ROOT/logs/nginx.log" >&2
}

# True when the marker names a live quantflow-app.sh owner process.
_marker_owner_live() {
    local pid=""
    [ -f "$MARKER" ] || return 1
    pid=$(sed -n 's/^pid=//p' "$MARKER" | head -n 1)
    case "$pid" in ''|*[!0-9]*) return 1 ;; esac
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "quantflow-app.sh" || return 1
    return 0
}

# True when the marker was left by --no-browser: ours, but with no owner
# window managing teardown (stop it later with --stop).
_marker_detached() {
    grep -q '^detached=1$' "$MARKER" 2>/dev/null
}

# ── --stop: only ever stops a stack this script started ──────────────────────

if [ "$DO_STOP" = "1" ]; then
    if [ -f "$MARKER" ] && ! _marker_detached && ! _marker_owner_live; then
        echo "Ignoring stale marker (owner process is gone); treating the stack as foreign."
        rm -f "$MARKER"
    fi
    if [ -f "$MARKER" ]; then
        if _stack_live; then
            echo "Stopping QuantFlow stack started by quantflow-app.sh..."
            bash "$SERVE_SH" --stop
        else
            echo "QuantFlow stack already stopped; clearing stale marker."
        fi
        rm -f "$MARKER"
        exit 0
    fi
    if _stack_live; then
        echo "Refusing to stop: this stack was NOT started by quantflow-app.sh," >&2
        echo "so it may belong to someone else (e.g. deer-flow-dev.service)." >&2
        if systemctl --user is-active deer-flow-dev.service >/dev/null 2>&1; then
            echo "deer-flow-dev.service is active; manage it with:" >&2
            echo "  systemctl --user stop deer-flow-dev.service" >&2
        else
            echo "Manage it from the checkout that started it ('make stop')." >&2
        fi
        exit 1
    fi
    echo "Nothing to stop: no QuantFlow stack is running."
    exit 0
fi

# ── Prerequisites (checked only when WE would start the stack) ───────────────

_check_prereqs() {
    local missing=0
    # Mirror serve.sh: explicit env path wins, then backend/, then repo root.
    if ! { { [ -n "${DEER_FLOW_CONFIG_PATH:-}" ] && [ -f "$DEER_FLOW_CONFIG_PATH" ]; } || \
           [ -f backend/config.yaml ] || [ -f config.yaml ]; }; then
        echo "✗ No DeerFlow config file found." >&2
        echo "  Run 'make setup' (recommended) or 'make config' to generate config.yaml." >&2
        missing=1
    fi
    if [ ! -f "$REPO_ROOT/.env" ]; then
        echo "✗ Missing $REPO_ROOT/.env." >&2
        echo "  Run 'make setup' to create it; quantflow-app.sh never generates configs or secrets." >&2
        missing=1
    fi
    if ! { { [ -n "${DEER_FLOW_EXTENSIONS_CONFIG_PATH:-}" ] && [ -f "$DEER_FLOW_EXTENSIONS_CONFIG_PATH" ]; } || \
           [ -f backend/extensions_config.json ] || [ -f extensions_config.json ]; }; then
        echo "✗ Missing extensions_config.json (repo root or backend/)." >&2
        echo "  Run 'make setup' to create it; quantflow-app.sh never generates configs or secrets." >&2
        missing=1
    fi
    return "$missing"
}

# ── Ensure stack ─────────────────────────────────────────────────────────────
# Sets STARTED_BY_ME=1 only when this invocation launches serve.sh.

_ensure_stack() {
    if _stack_live; then
        echo "Reusing already-running QuantFlow stack (not started by this script)."
        echo "This stack will be left running when the window closes."
        # A reused stack may still be mid-startup (e.g. service just restarted).
        if ! _wait_healthy 60 180 60; then
            echo "✗ Reused stack did not become healthy in time." >&2
            _print_log_paths
            return 1
        fi
        return 0
    fi

    _check_prereqs || return 1

    local mode="$MODE" serve_args=()
    if [ -z "$mode" ]; then
        if [ -f "$REPO_ROOT/frontend/.next/BUILD_ID" ]; then
            mode="prod"
        else
            mode="dev"
        fi
        echo "Auto mode: $mode (no --dev/--prod given)."
    fi
    serve_args=("--$mode" --daemon)
    if [ "$mode" = "prod" ]; then
        if [ -n "$SKIP_FRONTEND_BUILD" ] || [ -f "$REPO_ROOT/frontend/.next/BUILD_ID" ]; then
            serve_args+=(--skip-frontend-build)
        fi
    elif [ -n "$SKIP_FRONTEND_BUILD" ]; then
        echo "Note: --skip-frontend-build is ignored in dev mode (next dev does not build)."
    fi

    echo "Starting QuantFlow stack (${serve_args[*]})..."
    if ! bash "$SERVE_SH" "${serve_args[@]}"; then
        echo "✗ serve.sh failed to start the stack." >&2
        _print_log_paths
        return 1
    fi
    STARTED_BY_ME=1
    # --no-browser exits without an owner window, so its marker is detached
    # (ours, but ownerless) rather than owned by a live process.
    local detached=0
    [ "$NO_BROWSER" = "1" ] && detached=1
    printf 'pid=%s\nmode=%s\nstarted=%s\ndetached=%s\n' "$$" "$mode" "$(date -u +%FT%TZ)" "$detached" > "$MARKER"

    # serve.sh already waited per-service; this is a cheap backstop.
    if ! _wait_healthy 10 10 10; then
        echo "✗ Stack started but did not stay healthy; stopping what we started." >&2
        _print_log_paths
        bash "$SERVE_SH" --stop >/dev/null 2>&1 || true
        rm -f "$MARKER"
        STARTED_BY_ME=0
        return 1
    fi
    return 0
}

# ── Browser command ──────────────────────────────────────────────────────────

if [ -n "${QUANTFLOW_BROWSER_CMD:-}" ]; then
    # shellcheck disable=SC2206
    BROWSER_CMD=($QUANTFLOW_BROWSER_CMD)
elif [ "$NO_BROWSER" = "0" ]; then
    # Chromium is required only when we will actually open a window.
    if ! command -v chromium >/dev/null 2>&1; then
        echo "✗ chromium is not installed (needed for the QuantFlow app window)." >&2
        echo "  Install it, or set QUANTFLOW_BROWSER_CMD to an alternate command." >&2
        exit 1
    fi
    BROWSER_CMD=(chromium --app="$URL" --user-data-dir="$PROFILE")
fi

# ── Lock: only the owner traps teardown ──────────────────────────────────────

exec 9>"$LOCK"
if flock -n 9 2>/dev/null; then
    if ! _ensure_stack; then
        exit 1
    fi
    if [ "$NO_BROWSER" = "1" ]; then
        echo "QuantFlow is healthy: $URL (gateway :$GATEWAY_PORT, frontend :$FRONTEND_PORT, nginx :$NGINX_PORT)."
        if [ "$STARTED_BY_ME" = "1" ]; then
            echo "No owner window — this stack stays up until './quantflow-app.sh --stop'."
        fi
        exit 0
    fi
    teardown() {
        if [ "$STARTED_BY_ME" = "1" ]; then
            bash "$SERVE_SH" --stop >/dev/null 2>&1 || true
            rm -f "$MARKER"
        fi
    }
    trap teardown EXIT INT TERM
    "${BROWSER_CMD[@]}"
else
    # Extra window: another owner manages the stack; never touch it here.
    if [ "$NO_BROWSER" = "1" ]; then
        if ! _wait_healthy 60 180 60; then
            echo "✗ Stack did not become healthy in time." >&2
            _print_log_paths
            exit 1
        fi
        echo "QuantFlow is healthy: $URL (owned by another quantflow-app.sh instance)."
        exit 0
    fi
    if ! _wait_healthy 60 180 60; then
        echo "✗ Stack did not become healthy in time; not opening a window." >&2
        _print_log_paths
        exit 1
    fi
    echo "app already running — opening an extra window (first window owns shutdown)"
    setsid "${BROWSER_CMD[@]}" </dev/null >>"$LOCK_DIR/quantflow-extra.log" 2>&1 &
fi
