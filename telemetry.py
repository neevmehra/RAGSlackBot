from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Set the global tracer provider
trace.set_tracer_provider(TracerProvider())

# Add a simple span processor with console output
console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(console_exporter)
)

# Create and expose the tracer
tracer = trace.get_tracer(__name__)

# Add this missing function so app.py stops crashing
def setup_telemetry(app):
    FlaskInstrumentor().instrument_app(app)
    return tracer