# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# OCI imports
import oci
import os
import time
from dotenv import load_dotenv

# Load OCI environment variables
load_dotenv()

# ---- OpenTelemetry Setup ----
trace.set_tracer_provider(TracerProvider())
console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(console_exporter))
tracer = trace.get_tracer(__name__)

def setup_telemetry(app):
    FlaskInstrumentor().instrument_app(app)
    return tracer

# ---- OCI Monitoring Setup ----
# oci_config = {
#     "user": os.getenv("OCI_USER_OCID"),
#     "key_file": os.getenv("OCI_KEY_FILE"),
#     "fingerprint": os.getenv("OCI_FINGERPRINT"),
#     "tenancy": os.getenv("OCI_TENANCY_OCID"),
#     "region": os.getenv("OCI_REGION"),
# }

# monitoring_client = oci.monitoring.MonitoringClient(oci_config)
# compartment_id = os.getenv("OCI_COMPARTMENT_OCID")

def push_custom_metric(value: float, metric_name="slackbot_latency_ms"):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metric_data = oci.monitoring.models.MetricDataDetails(
        namespace="custom_metrics",
        resource_group="llmapp",
        name=metric_name,
        dimensions={"service": "slackbot"},
        datapoints=[
            oci.monitoring.models.Datapoint(
                timestamp=timestamp,
                value=value,
                count=1
            )
        ],
        metadata={"unit": "ms"}
    )
    request = oci.monitoring.models.PutMetricDataDetails(metric_data=[metric_data])
    response = monitoring_client.put_metric_data(request, compartment_id=compartment_id)
    print(f"[✅] Metric '{metric_name}' pushed: {response.status}")
