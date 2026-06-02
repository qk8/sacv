#!/usr/bin/env bash
# sandbox-start.sh
# =================
# Starts background services inside the Docker sandbox container.
# Runs on container startup via CMD; keeps the container alive for docker exec.

set -euo pipefail

echo "[sacv-sandbox] Starting background services..."

# ── Start Jaeger all-in-one (OTel collector + query) ─────────────────────────
if command -v jaeger &>/dev/null; then
    SPAN_STORAGE_TYPE=memory jaeger \
        --collector.otlp.grpc.host-port=0.0.0.0:4317 \
        --collector.otlp.http.host-port=0.0.0.0:4318 \
        --query.http-server.host-port=0.0.0.0:16686 \
        &>/tmp/jaeger.log &
    echo "[sacv-sandbox] Jaeger started (OTLP :4317/:4318, query :16686)"
else
    echo "[sacv-sandbox] Jaeger not found — OTel trace correlation disabled"
fi

# ── Wait for ports to be ready ────────────────────────────────────────────────
sleep 1

# Signal that the container is ready to receive docker exec commands.
# DockerContainerManager._wait_for_ready() polls for this file.
touch /tmp/sacv-ready

# ── Attach OTel Java agent to all JVM processes ───────────────────────────
if [ -f /opt/opentelemetry-javaagent.jar ]; then
    export JAVA_TOOL_OPTIONS="\
        -javaagent:/opt/opentelemetry-javaagent.jar \
        -Dotel.service.name=sacv-sandbox \
        -Dotel.exporter.otlp.endpoint=http://localhost:4317 \
        -Dotel.exporter.otlp.protocol=grpc \
        -Dotel.traces.exporter=otlp"
    echo "[sacv-sandbox] OTel Java agent attached (OTLP :4317)"
else
    echo "[sacv-sandbox] OTel agent not found — Java tracing disabled"
fi

echo "[sacv-sandbox] Ready. Waiting for docker exec commands..."
tail -f /dev/null
