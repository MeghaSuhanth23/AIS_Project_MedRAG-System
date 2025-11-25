from prometheus_client import start_http_server
import threading

def start_metrics_server(port=8000):
    """Starts the Prometheus metrics endpoint in the background."""
    def run():
        start_http_server(port)  # exposes /metrics
        
    thread = threading.Thread(target=run, daemon=True)
    thread.start()