from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY


def get_or_create_counter(name, description, labelnames=None):
    """Get existing counter or create new one"""
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Counter(name, description, labelnames or [])

def get_or_create_histogram(name, description, labelnames=None, buckets=None):
    """Get existing histogram or create new one"""
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        if buckets:
            return Histogram(name, description, labelnames or [], buckets=buckets)
        return Histogram(name, description, labelnames or [])

def get_or_create_gauge(name, description, labelnames=None):
    """Get existing gauge or create new one"""
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Gauge(name, description, labelnames or [])

feedback_counter = get_or_create_counter(
    'medrag_feedback_total',
    'Total feedback responses from users',
    ['feedback_type']  # 'positive' or 'negative'
)

feedback_response_time = get_or_create_histogram(
    'medrag_feedback_response_seconds',
    'Time taken by user to provide feedback after receiving answer',
    buckets=[5, 10, 30, 60, 120, 300, 600]  # 5s to 10min
)

query_counter = get_or_create_counter(
    'medrag_queries_total',
    'Total number of queries processed',
    ['status']  
)

query_duration = get_or_create_histogram(
    'medrag_query_duration_seconds',
    'Time taken to process a query',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Confidence score distribution
confidence_scores = get_or_create_histogram(
    'medrag_confidence_scores',
    'Distribution of similarity confidence scores',
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
)

# Number of documents retrieved per query
documents_retrieved = get_or_create_histogram(
    'medrag_documents_retrieved',
    'Number of relevant documents retrieved per query',
    buckets=[0, 1, 2, 3, 4, 5, 10, 20]
)

# Specialty distribution
specialty_queries = get_or_create_counter(
    'medrag_specialty_queries_total',
    'Queries by medical specialty',
    ['specialty']
)


# API errors by type
api_errors = get_or_create_counter(
    'medrag_api_errors_total',
    'Total API errors encountered',
    ['error_type']
)

# Gemini API calls
gemini_api_calls = get_or_create_counter(
    'medrag_gemini_calls_total',
    'Total calls to Gemini API',
    ['status']  # 'success' or 'failed'
)

# Active sessions gauge
active_sessions = get_or_create_gauge(
    'medrag_active_sessions',
    'Number of active user sessions'
)

# Vector DB size
vector_db_size = get_or_create_gauge(
    'medrag_vector_db_documents',
    'Total documents in vector database'
)

# Model load time
model_load_time = get_or_create_histogram(
    'medrag_model_load_seconds',
    'Time taken to load models and vector DB',
    buckets=[1, 5, 10, 30, 60, 120]
)


# Low confidence query counter
low_confidence_queries = get_or_create_counter(
    'medrag_low_confidence_queries_total',
    'Queries that returned low confidence results (<0.7)'
)

# High confidence query counter
high_confidence_queries = get_or_create_counter(
    'medrag_high_confidence_queries_total',
    'Queries that returned high confidence results (>0.85)'
)

# Citation count per response
citations_per_response = get_or_create_histogram(
    'medrag_citations_per_response',
    'Number of citations included in responses',
    buckets=[0, 1, 2, 3, 4, 5, 10]
)

# PDF exports
pdf_exports = get_or_create_counter(
    'medrag_pdf_exports_total',
    'Total PDF exports requested',
    ['status']  # 'success' or 'failed'
)

# Chat history views
chat_history_views = get_or_create_counter(
    'medrag_chat_history_views_total',
    'Number of times chat history was accessed'
)

# Statistics dashboard views
stats_dashboard_views = get_or_create_counter(
    'medrag_stats_dashboard_views_total',
    'Number of times statistics dashboard was accessed'
)