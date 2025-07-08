from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set the global tracer provider
trace.set_tracer_provider(TracerProvider())

# Add a simple span processor with console output
console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(console_exporter)
)

# Create and expose the tracer
tracer = trace.get_tracer(__name__)
