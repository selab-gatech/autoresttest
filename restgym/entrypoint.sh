#!/bin/bash
# Starts the local model server, writes AutoRestTest's configuration from the
# variables RESTgym sets (API, HOST, PORT, TIME_BUDGET), then runs AutoRestTest
# until RESTgym stops the container.

HOST="${HOST:-localhost}"
PORT="${PORT:-8080}"
TIME_BUDGET="${TIME_BUDGET:-60}"
MODEL_ALIAS="gemma-4-26b-a4b"

# The container shares the host's network, so ask the OS for a free port.
LLM_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')

# llama-server finds its libraries only when started from /app.
DEVICES=$(cd /app && timeout 60 ./llama-server --list-devices 2>&1)
echo "$DEVICES"
LLM_PID=""
if grep -qE '^ +CUDA[0-9]+:' <<< "$DEVICES"; then
    # 8 slots match AutoRestTest's 8 value-generation workers and share one 64k-token KV cache.
    # --cache-ram 0 and --ctx-checkpoints 0 keep prompt caches out of the 16 GB container RAM limit.
    # --threads stays small because the host may report far more cores than the container's 8-CPU quota.
    (cd /app && exec ./llama-server \
        --model "$MODEL_PATH" \
        --alias "$MODEL_ALIAS" \
        --host 127.0.0.1 \
        --port "$LLM_PORT" \
        --flash-attn on \
        --ctx-size 65536 \
        --parallel 8 \
        --kv-unified \
        --cache-ram 0 \
        --ctx-checkpoints 0 \
        --reasoning off \
        --threads 4) &
    LLM_PID=$!
else
    # On the CPU, the model would take most of the hour to generate values.
    echo "No CUDA device found; not starting the model server."
fi

cat > /tool/configurations.toml <<EOF
[spec]
location = "/specifications/${API}-openapi.json"
recursion_limit = 1
strict_validation = false

[llm]
engine = "${MODEL_ALIAS}"
api_base = "http://127.0.0.1:${LLM_PORT}/v1"
creative_temperature = 1.0
strict_temperature = 1.0
max_tokens = 4096
timeout_seconds = 300.0

[agent]
max_combinations = 12
max_total_combinations = 3000
base_samples_per_size = 200
combination_seed = 42

[agent.value]
parallelize = true
max_workers = 8

[agents.header]
enabled = false

[cache]
# Each run starts with an empty cache, so this only lets a restart skip setup.
use_cached_graph = true
use_cached_table = true

[q_learning]
learning_rate = 0.1
discount_factor = 0.9
max_exploration = 1.0

[request_generation]
time_duration = $((TIME_BUDGET * 60))
mutation_rate = 0.2

[api]
override_url = true
host = "${HOST}"
port = ${PORT}
request_timeout_seconds = 30.0
EOF

echo "AutoRestTest configured: API=${API} target=${HOST}:${PORT} budget=${TIME_BUDGET}m llm=127.0.0.1:${LLM_PORT}"

# Wait up to 5 minutes for the model to load. If it never becomes ready,
# AutoRestTest still runs, using its non-LLM value sources.
llm_ready() { curl -sf "http://127.0.0.1:${LLM_PORT}/health" > /dev/null; }
if [ -n "$LLM_PID" ]; then
    for _ in $(seq 150); do
        if llm_ready || ! kill -0 "$LLM_PID" 2> /dev/null; then
            break
        fi
        sleep 2
    done
fi
if llm_ready; then
    echo "Model server ready."
else
    echo "Model server not ready; AutoRestTest will run without LLM-generated values."
fi

# Restart AutoRestTest if it ever exits, since an exited container counts as a failed run.
cd /tool || exit 1
while true; do
    autoresttest --skip-wizard
    sleep 1
done
