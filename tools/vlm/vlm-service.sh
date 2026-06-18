#!/bin/bash

# VLM Service Manager with Auto-restart
# Usage: ./vlm-service.sh --root <llama.cpp_dir> -m <model_path> --mmproj <mmproj_path> --port <port>

# Default values
ROOT="/home/workspace/source/llama.cpp"
MODEL_PATH="/home/data/Qwen/Qwen3-VL-8B-Instruct-F16.gguf"
MMPROJ_PATH="/home/data/Qwen/Qwen3-VL-8B-Instruct-F16-MMPROJ-F16.gguf"
PORT=8080
LLAMA_SERVER_RELATIVE="build/bin/llama-server"
MEDIA_PATH="/home/workspace/source/temp"
CTX_SIZE=262144
PARALLEL=32
FLASH_ATTN="off"
CHECK_INTERVAL=1

# 修复：由于 fd 耗尽（1024 个），导致的 failed to launch ffprobe 问题。
ulimit -n 1048576
# export CUDA_VISIBLE_DEVICES=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root)
            ROOT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mmproj)
            MMPROJ_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --media-path)
            MEDIA_PATH="$2"
            shift 2
            ;;
        -c|--ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        -np|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -fa|--flash-attn)
            FLASH_ATTN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --root PATH         llama.cpp working directory (default: $ROOT)"
            echo "  -m, --model PATH    Model file path (default: $MODEL_PATH)"
            echo "  --mmproj PATH       MMPROJ file path (default: $MMPROJ_PATH)"
            echo "  --port PORT         Service port (default: $PORT)"
            echo "  --media-path PATH   Media files directory (default: $MEDIA_PATH)"
            echo "  -c, --ctx-size N    Context size (default: $CTX_SIZE)"
            echo "  -np, --parallel N   Parallel slots (default: $PARALLEL)"
            echo "  -fa, --flash-attn V Flash attention: on|off|auto (default: $FLASH_ATTN)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information." >&2
            exit 1
            ;;
    esac
done

# Resolve llama-server path relative to ROOT
LLAMA_SERVER="$ROOT/$LLAMA_SERVER_RELATIVE"

# Validate required files and directories
if [[ ! -d "$ROOT" ]]; then
    echo "[ERROR] ROOT directory not found: $ROOT" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model file not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$MMPROJ_PATH" ]]; then
    echo "[ERROR] MMPROJ file not found: $MMPROJ_PATH" >&2
    exit 1
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "[ERROR] llama-server not found or not executable: $LLAMA_SERVER" >&2
    exit 1
fi

# Track current child PID for the server bound to our port
CHILD_PID=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [port=$PORT] $*"
}

# Find PID of llama-server process listening on our specific port.
# This isolates our process from other instances using different ports.
find_server_pid_by_port() {
    local pid=""
    if command -v ss >/dev/null 2>&1; then
        pid=$(ss -lntpH "sport = :$PORT" 2>/dev/null \
            | grep -oP 'pid=\K[0-9]+' | head -n1)
    fi
    if [[ -z "$pid" ]] && command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1)
    fi
    echo "$pid"
}

# Check whether the tracked child process is alive.
# Falls back to a port-based lookup if the recorded PID is gone.
is_server_alive() {
    if [[ "$CHILD_PID" -gt 0 ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        return 0
    fi
    local port_pid
    port_pid=$(find_server_pid_by_port)
    if [[ -n "$port_pid" ]]; then
        CHILD_PID="$port_pid"
        return 0
    fi
    return 1
}

start_server() {
    # Refuse to start if another process is already listening on our port,
    # unless it is the child we previously spawned.
    local existing_pid
    existing_pid=$(find_server_pid_by_port)
    if [[ -n "$existing_pid" && "$existing_pid" != "$CHILD_PID" ]]; then
        log "Port $PORT already in use by PID $existing_pid; adopting it."
        CHILD_PID="$existing_pid"
        return 0
    fi

    # If run on nvidia 3090x2, change the parameters to:
    # --ctx-size 131072
    # --parallel 4
    # -fa on
    log "Starting llama-server (root=$ROOT, model=$MODEL_PATH, port=$PORT, ctx=$CTX_SIZE, parallel=$PARALLEL, fa=$FLASH_ATTN)"
    # Use setsid so llama-server becomes its own session/process group leader.
    # This lets us send signals to the entire process group during cleanup,
    # killing llama-server and any helper processes (e.g. ffprobe) it spawned.
    # If setpriv is available, use --pdeathsig TERM so the kernel kills the
    # child even when the parent is hit by SIGKILL (which cannot be trapped).
    local launcher=()
    if command -v setpriv >/dev/null 2>&1; then
        launcher=(setpriv --pdeathsig TERM --)
    fi
    local server_args=(
        -m "$MODEL_PATH"
        --mmproj "$MMPROJ_PATH"
        -n 8192 -c "$CTX_SIZE" -ngl 99
        -b 8192 --ubatch-size 4096
        -fa "$FLASH_ATTN"
        -np "$PARALLEL"
        --media-path "$MEDIA_PATH"
        --host 0.0.0.0 --port "$PORT"
    )
    log "Command: (cd $ROOT && setsid ${launcher[*]} $LLAMA_SERVER ${server_args[*]})"
    (
        cd "$ROOT" || exit 1
        exec setsid "${launcher[@]}" "$LLAMA_SERVER" "${server_args[@]}"
    ) &
    CHILD_PID=$!
    log "Spawned llama-server PID=$CHILD_PID"
}

cleanup() {
    log "Shutdown requested, stopping server PID=$CHILD_PID"
    if [[ "$CHILD_PID" -gt 0 ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        # Send TERM to the entire process group to catch llama-server
        kill -TERM -"$CHILD_PID" 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null || true
        # Give it a few seconds to exit cleanly
        for _ in 1 2 3 4 5; do
            kill -0 "$CHILD_PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still alive
        kill -KILL -"$CHILD_PID" 2>/dev/null || kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}

# Trap multiple signals to ensure cleanup on various termination scenarios
trap cleanup INT TERM EXIT HUP QUIT

# Initial start
start_server

# Monitor loop: check every CHECK_INTERVAL seconds, restart only if our
# port-bound process is gone (so we never disturb other instances).
while true; do
    sleep "$CHECK_INTERVAL"
    if ! is_server_alive; then
        log "Server on port $PORT is down (last PID=$CHILD_PID), restarting..."
        CHILD_PID=0
        start_server
    fi
done
