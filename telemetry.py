import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from prometheus_client import start_http_server

# ---- Tracing ----
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer(__name__)

# ---- Metrics globals ----
_meter = None
_latency_hist = None
_error_counter = None
_hist_cache = {}

def setup_telemetry(app, service_name="supportagent", service_version="1.0.0", prom_port=9464):
    """Exposes Prometheus metrics and instruments the Flask app for telemetry.

    Sets up an OpenTelemetry MeterProvider and exports metrics on the given port (default 9464). It also instruments the Flask application and the requests library for tracing, and retains console tracing via the SimpleSpanProcessor.
    """
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": os.getenv("ENV", "dev"),
    })

    # Metrics (Prometheus)
    reader = PrometheusMetricReader()  # integrates with prometheus_client
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    global _meter, _latency_hist, _error_counter
    _meter = metrics.get_meter(service_name)

    # Histogram for latency
    _latency_hist = _meter.create_histogram(
        name="oraclebot_latency_ms",
        description="Latency of oraclebot responses (ms)",
        unit="ms",
    )

    # Counter for errors
    _error_counter = _meter.create_counter(
        name="oraclebot_errors_total",
        description="Count of oraclebot errors",
        unit="1",
    )

    # Start /metrics HTTP endpoint (non-blocking)
    start_http_server(prom_port)

    # Auto-instrument Flask and outgoing HTTP calls
    FlaskInstrumentor().instrument_app(app)
    RequestsInstrumentor().instrument()

    return tracer

def push_custom_metric(value: float, metric_name="oraclebot_latency_ms", success: bool = True):
    """Records a latency value in the histogram and increments the error counter if needed."""
    global _hist_cache, _meter
    if metric_name not in _hist_cache and _meter:
        _hist_cache[metric_name] = _meter.create_histogram(
            name=metric_name,
            description=f"{metric_name} (ms)",
            unit="ms",
        )
    hist = _hist_cache.get(metric_name)
    if hist:
        hist.record(value)

    if not success and _error_counter:
        _error_counter.add(1, attributes={"metric_name": metric_name})