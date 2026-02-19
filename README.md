# MED RAG: System for Evidence-Based Clinical Decisions

## Project Overview

**MED RAG** is an end-to-end Retrieval-Augmented Generation (RAG) system designed to revolutionize access to medical literature. By combining BioBERT embeddings, FAISS vector search, and Google's Gemini AI, the system enables healthcare professionals and researchers to query **5,000+ peer-reviewed medical articles** across **8 specialties** with high accuracy and confidence scoring.

### Problem Statement

Healthcare professionals face critical barriers when accessing medical literature:
- **Information Overload:** 35+ million citations in PubMed make manual search inefficient
- **Semantic Gap:** Traditional keyword matching misses semantically related research
- **Fragmented Results:** No contextual synthesis across multiple papers
- **Time Burden:** Hours of manual reading delay clinical decisions

### Solution

MED RAG addresses these challenges by:
- Reducing evidence search time from **hours to <1 second**
- Achieving **92% retrieval accuracy** with semantic understanding
- Providing **100% citation traceability** to PubMed sources
- Delivering contextual answers with **68% high-confidence results (>0.85)**

---

## Repository Contents
```
MED_RAG/
├── src/                                    # Main source code
│   ├── main.py                             # Streamlit application (main entry point)
│   ├── requirements.txt                    # Python dependencies
│   ├── metrics_server.py                   # Prometheus metrics server
│   │
│   ├── utils/                              # Utility modules
│   │   ├── feedback_utils.py               # Feedback system logic
│   │   ├── feedback_metrics.py             # Feedback analytics
│   │   ├── rag_pipeline.py                 # RAG orchestration
│   │   └── vector_db_utils.py              # FAISS operations
│   │
│   ├── data_specialities_checkpoints/      # Specialty-specific datasets
│   │   ├── cardiology_checkpoint.csv
│   │   ├── diabetes_checkpoint.csv
│   │   ├── gastroenterology_checkpoint.csv
│   │   ├── infectious_diseases_checkpoint.csv
│   │   ├── nephrology_checkpoint.csv
│   │   ├── neurology_checkpoint.csv
│   │   ├── oncology_checkpoint.csv
│   │   └── pulmonology_checkpoint.csv
│   │
│   ├── embeddings/                         # Pre-generated BioBERT embeddings
│   │   ├── complete_20251122_210039.pkl
│   │   ├── embeddings_20251122_210039.npy
│   │   └── metadata_20251122_210039.csv
│   │
│   ├── models/vector_database/             # FAISS index files
│   │   └── [FAISS indices and metadata]
│   │
│   ├── exported_chats/                     # PDF exports storage
│   ├── videos/                             # Demo videos
│   │   └── MedRAG_Visualization_Video.mp4
│   │
│   ├── trustworthiness_data/               # Model reliability metrics
│   │   ├── trustworthiness_analysis_data.json
│   │   └── trustworthiness_report_data.txt
│   │
│   ├── trustworthiness_model/              # Trust analysis models
│   │   ├── trustworthiness_analysis_model.json
│   │   └── trustworthiness_report_model.txt
│   │
│   ├── risk_management_data/               # Risk assessment data
│   │   ├── risk_analysis_data.json
│   │   └── risk_report_data.txt
│   │
│   ├── risk_management_model/              # Risk models
│   │   ├── risk_analysis_model.json
│   │   └── risk_report_model.txt
│   │
│   └── .gitattributes                      # Git LFS configuration
│
├── medical_literature_dataset.csv          # Complete dataset
├── medical_literature_dataset.json         # Dataset (JSON format)
├── demo_results_20251122_213407.json       # Sample demo results
├── chat_history.json                       # Persistent chat storage
├── feedback.jsonl                          # Feedback logs
├── playground_MedRAG.ipynb                 # **THIS IPYNB FILE IS JUST FOR REFERENCE**
├── Total performance Report.txt            # Performance metrics
│
├── deployment/                             # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── monitoring/                             # Observability stack
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── medrag_dashboard.json
│
├── model_visualizations/                   # Analysis charts
│   ├── Accuracy_breakdown.png
│   ├── Citation_analysis.png
│   ├── dataset_composition.png
│   ├── Latency_timeline.png
│   └── Speciality_performance.png
│
├── documentation/
│   ├── README.md
│   └── AIS Report Evaluation.pdf
│
├── .streamlit/
│   └── config.toml
│
└── .gitignore
---

## System Entry Point

### Main Script: `src/main.py`

The application is built with **Streamlit** and serves as both the frontend UI and backend processing engine.

### Running Locally

#### Prerequisites
- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

#### Installation
```bash
# Clone the repository
git clone https://github.com/MeghaSuhanth23/AIS_Project_MedRAG-System.git
cd MED_RAG

# Install dependencies
pip install -r src/requirements.txt

# Navigate to src directory
cd src

# Run the application
streamlit run main.py
```

The application will be available at `http://localhost:8501`

#### First-Time Setup

1. **Enter API Key:** On first launch, enter Gemini API key
2. **Initialize System:** Click "Initialize System" to load models
3. **Start Querying:** Ask medical questions and receive evidence-based answers

---

## Deployment Strategy

### Docker Containerization

The system is fully containerized for consistent deployment across environments.

#### Build and Run with Docker
```bash
# Build the Docker image
docker build -t medrag-system -f deployment/Dockerfile .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/src/models/vector_database:/app/models/vector_database \
  -v $(pwd)/src/embeddings:/app/embeddings \
  -v $(pwd)/src/exported_chats:/app/exported_chats \
  medrag-system
```

#### Using Docker Compose (Recommended)
```bash
# Start all services (app + monitoring)
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.yml down
```

#### Docker Compose Services

- **medrag-app:** Main Streamlit application (port 8501)
- **prometheus:** Metrics collection (port 9090)
- **grafana:** Monitoring dashboards (port 3000)

### Production Deployment

For production environments:
```bash
# Run in detached mode
docker-compose -f deployment/docker-compose.yml up -d

# Check service health
docker-compose -f deployment/docker-compose.yml ps

# Scale if needed (future)
docker-compose -f deployment/docker-compose.yml up -d --scale medrag-app=3
```

---

## Monitoring and Metrics

### Tools Used

- **Prometheus:** Time-series metrics collection
- **Grafana:** Real-time visualization dashboards
- **Custom Metrics Server:** `src/metrics_server.py` exposes application metrics

### Monitored Metrics

1. **Performance Metrics**
   - Query response time (P50, P95, P99)
   - FAISS search latency
   - Gemini API response time
   - Total queries processed

2. **Quality Metrics**
   - Confidence score distribution
   - Retrieval accuracy per specialty
   - User feedback ratings (thumbs up/down)
   - Citation coverage percentage

3. **System Metrics**
   - Container resource usage (CPU, memory)
   - Request rate and throughput
   - Error rate and types
   - Cache hit ratio

### Accessing Monitoring Dashboards

**Prometheus:**
```
http://localhost:9090
```

**Grafana:**
```
http://localhost:3000
Default credentials: admin/admin
```

**Import Dashboard:**
1. Login to Grafana
2. Go to Dashboards → Import
3. Upload `monitoring/grafana/medrag_dashboard.json`

### Metrics Endpoint

Custom metrics are exposed at:
```
http://localhost:8000/metrics
```

---

##  Video Demonstration

**Full System Demo:** [`videos/MedRAG_Visualization_Video.mp4`](videos/MedRAG_Visualization_Video.mp4)


## Project Documentation

### Key Documents

1. **[AI System Project Report](documentation/AIS%20Report%20Evaluation.pdf)**
   - Complete system documentation
   - Performance evaluation
   - Risk and trustworthiness analysis

2. **Model Visualizations:**
   - [Accuracy Breakdown](model_visualizations/Accuracy_breakdown.png)
   - [Citation Analysis](model_visualizations/Citation_analysis.png)
   - [Dataset Composition](model_visualizations/dataset_composition.png)
   - [Latency Timeline](model_visualizations/Latency_timeline.png)
   - [Specialty Performance](model_visualizations/Speciality_performance.png)

3. **Risk Management Reports:**
   - Risk analysis data: `risk_management_data/risk_analysis_data.json`
   - Risk report: `risk_management_data/risk_report_data.txt`

4. **Trustworthiness Evaluation:**
   - Trust metrics: `trustworthiness_data/trustworthiness_analysis_data.json`
   - Evaluation report: `trustworthiness_data/trustworthiness_report_data.txt`

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.31.0 |
| **Embeddings** | BioBERT | dmis-lab/biobert-v1.1 |
| **Vector Search** | FAISS | 1.7.4 |
| **LLM** | Google Gemini | 1.5 Flash |
| **Backend** | Python | 3.10+ |
| **PDF Export** | ReportLab | 4.0.9 |
| **Monitoring** | Prometheus + Grafana | Latest |
| **Containerization** | Docker | Latest |
| **Orchestration** | Docker Compose | 3.8 |

---

##  System Performance

| Metric | Result |
|--------|--------|
| **Retrieval Accuracy** | 92% |
| **Average Response Time** | 0.8 seconds |
| **High Confidence Queries** | 68% (>0.85) |
| **User Satisfaction** | 4.2/5 ⭐ |
| **Citation Coverage** | 100% |
| **Dataset Size** | 5,000+ papers |
| **Specialties Covered** | 8 medical domains |
| **Embedding Dimension** | 768 (BioBERT) |

---

##  Configuration

### Environment Variables

Create a `.env` file in the root directory:
```env
# Gemini API Configuration
GEMINI_API_KEY=AIzaSyA-1aB37Aea_JRxYNdClwfS4R5ib7v8Vkg

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
METRICS_SERVER_PORT=8000
```

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize UI:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#f5f7fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#2d3748"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
```

---

##  Development & Testing

### Development Environment
```bash
# Install development dependencies
pip install -r src/requirements.txt

```

### Testing Queries

Use `demo_results_20251122_213407.json` for sample queries and expected outputs.

---

##  Data Management

### Dataset Information

- **Source:** PubMed Central (NCBI)
- **Size:** 5,000+ peer-reviewed papers
- **Format:** CSV and JSON
- **Time Range:** 2020-2024
- **Specialties:** Diabetes, Cardiology, Infectious Diseases, Neurology, Pulmonology, Oncology, Nephrology, Gastroenterology

### Specialty Checkpoints

Individual specialty data is stored in `src/data_specialties_checkpoints/`:
- `cardiology_checkpoint.csv`
- `diabetes_checkpoint.csv`
- `gastroenterology_checkpoint.csv`
- `infectious_diseases_checkpoint.csv`
- `nephrology_checkpoint.csv`
- `neurology_checkpoint.csv`
- `oncology_checkpoint.csv`
- `pulmonology_checkpoint.csv`

### Embeddings

Pre-computed BioBERT embeddings are stored in `src/embeddings/`:
- `complete_20251122_210039.pkl` - Complete embedding data structure
- `embeddings_20251122_210039.npy` - NumPy array (768-dim vectors)
- `metadata_20251122_210039.csv` - Document metadata mapping

---

## Security & Privacy

- **API Key Security:** Gemini API keys stored in environment variables
- **Docker Isolation:** Containerized deployment limits attack surface
- **HTTPS Ready:** Production deployment supports TLS encryption
- **User Privacy:** Feedback data anonymized and aggregated

---

## Version Control and Team Collaboration

### Git Workflow

- **Main Branch:** Production-ready code
- **Development Branch:** Active development
- **Feature Branches:** Individual features (`feature/feedback-system`, `feature/pdf-export`)
- **Hotfix Branches:** Critical bug fixes

### Code Review Process

1. Feature development in separate branch
2. Pull request with detailed description
3. Peer review and testing
4. Merge to development
5. Production deployment from main

### Collaboration Tools

- **GitHub:** Version control and issue tracking
- **Docker:** Consistent development environments
- **Jupyter:** Shared experimentation notebooks

### Commit Conventions
```
feat: Add new feature
fix: Bug fix
docs: Documentation update
refactor: Code refactoring
test: Add tests
chore: Maintenance tasks
```

---

## Known Limitations

1. **Specialty Coverage:** Currently limited to 8 medical specialties
2. **Language:** English-only papers
3. **Dataset Size:** 5,000+ papers (expandable to 50,000+)
4. **API Dependency:** Requires active Gemini API key
5. **Real-time Updates:** Dataset requires manual updates

---

## Future Enhancements

- [ ] Expand to 20+ medical specialties
- [ ] Add multi-modal support (medical images, clinical trials)
- [ ] Implement real-time PubMed API integration
- [ ] Deploy on cloud platforms (AWS, GCP, Azure)
- [ ] Add multi-language support
- [ ] Add authentication and user management

---


## Contributors

- **Megha Suhanth Royal Sarvepalli**
- **Course:** Masters in AIS
- **Institution:** University of Florida

---

## Acknowledgments

- **Google AI** - Gemini API access and documentation
- **Hugging Face** - BioBERT models and transformers library
- **NCBI/PubMed** - Medical literature database
- **Streamlit** - UI framework
- **FAISS Team** - Vector search library
- **Course Instructors** - Guidance and support

---

## Contact & Support

For questions or issues:
- **Email:** msarvepalli@ufl.edu

---

## Quick Start Guide
```bash
# 1. Clone repository
git clone https://github.com/MeghaSuhanth23/AIS_Project_MedRAG-System.git && cd MED_RAG

# 2. Set up environment
echo "GEMINI_API_KEY=AlzaSyA-1aB37Aea_JRxYNdClwfS4R5ib7v8Vkg" .env

# 3. Run with Docker
docker-compose -f deployment/docker-compose.yml up -d

# 4. Access application
open http://localhost:8501

# 5. View monitoring
open http://localhost:3000
```

---

**Built for evidence-based medicine**
