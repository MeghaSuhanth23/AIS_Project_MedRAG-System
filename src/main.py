import streamlit as st
import os
from datetime import datetime
import json
import time
from prometheus_client import start_http_server
from metrics_server import start_metrics_server
import threading

if "metrics_started" not in st.session_state:
    st.session_state.metrics_started = True
    threading.Thread(target=start_metrics_server, args=(8000,), daemon=True).start()


st.set_page_config(
    page_title="MED RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_metrics_server():

    try:
        start_http_server(port=8000, addr='0.0.0.0')
        return True
    except OSError:
        print("⚠️ Metrics server already running on port 8000")
        return True


init_metrics_server()

from utils.vector_db_utils import MedicalVectorDB
from utils.rag_pipeline import MedicalRAGPipeline

from utils.feedback_metrics import (
    query_counter, 
    query_duration,
    confidence_scores,
    api_errors,
    documents_retrieved,
    specialty_queries,
    high_confidence_queries,
    low_confidence_queries,
    citations_per_response,
    pdf_exports,
    model_load_time,
    feedback_counter  
)


DEFAULT_API_KEY = "AIzaSyCEYeOGh0ebEOjb3vCk8KOv3XTc55IkUxk"

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stTextInput input, .stTextArea textarea {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 2px solid #667eea !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-size: 1.1rem;
        padding: 0.7rem 2.5rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a2e !important;
        color: #eaeaea !important;
    }
    
    p, span, div, label {
        color: #eaeaea !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #262730 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 0.5rem;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="column"] button[key^="stat_"] {
        background: linear-gradient(135deg, #262730 0%, #16213e 100%) !important;
        color: #fafafa !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border: 1px solid #667eea !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        height: 150px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        white-space: pre-line !important;
    }

    [data-testid="column"] button[key^="stat_"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 12px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .stAlert {
        background-color: #262730 !important;
        color: #eaeaea !important;
    }
</style>
""", unsafe_allow_html=True)


def load_chat_history():
    """Load chat history from JSON file"""
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    return []


def save_chat_history(history):
    """Save chat history to JSON file with numpy type conversion"""
    import numpy as np
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_history = convert_to_serializable(history)
    
    with open('chat_history.json', 'w') as f:
        json.dump(serializable_history, f, indent=2)


def clean_html_tags(text):
    import re
    clean_text = re.sub(r'<[^>]+>', '', str(text))
    return clean_text


def export_to_pdf(chat_history):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER
    
    os.makedirs('exported_chats', exist_ok=True)
    
    filename = f"exported_chats/chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    doc = SimpleDocTemplate(filename, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#667eea',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#333333',
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    small_style = ParagraphStyle(
        'CustomSmall',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor='#666666'
    )

    def clean_text(text):
        if not text:
            return ""
        text = str(text)
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')
        return text
    
    story.append(Paragraph("MED RAG System - Chat Export", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style))
    story.append(Spacer(1, 0.3*inch))
    
    for i, chat in enumerate(chat_history, 1):
        try:
            story.append(Paragraph(f"<b>Query {i}:</b>", heading_style))
            query_text = clean_text(chat.get('query', 'No query'))
            story.append(Paragraph(query_text, normal_style))
            
            timestamp = chat.get('timestamp', 'N/A')
            story.append(Paragraph(f"<i>Time: {timestamp}</i>", small_style))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Answer:</b>", heading_style))
            answer_text = clean_text(chat.get('answer', 'No answer'))
            
            if len(answer_text) > 2000:
                answer_text = answer_text[:2000] + "... (truncated for PDF)"
            
            story.append(Paragraph(answer_text, normal_style))
            story.append(Spacer(1, 0.1*inch))
            
            sources = chat.get('sources', [])
            if sources:
                story.append(Paragraph(f"<b>Sources ({len(sources)}):</b>", heading_style))
                
                for j, source in enumerate(sources[:3], 1):
                    try:
                        pmid = source.get('pmid', 'N/A')
                        title = clean_text(source.get('title', 'No title'))
                        specialty = source.get('specialty', 'N/A')
                        
                        source_text = f"{j}. <b>PMID:</b> {pmid}<br/>"
                        source_text += f"<b>Title:</b> {title}<br/>"
                        source_text += f"<b>Specialty:</b> {specialty}"
                        
                        story.append(Paragraph(source_text, small_style))
                        story.append(Spacer(1, 0.05*inch))
                    except:
                        story.append(Paragraph(f"{j}. [Source information unavailable]", small_style))
                
                story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("_" * 80, small_style))
            story.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            story.append(Paragraph(f"<i>[Entry {i} could not be exported]</i>", small_style))
            story.append(Spacer(1, 0.2*inch))
            continue
    
    try:
        doc.build(story)
        return filename
    except Exception as e:
        simple_filename = f"exported_chats/simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        simple_doc = SimpleDocTemplate(simple_filename, pagesize=letter)
        simple_story = []
        
        simple_story.append(Paragraph("MED RAG System - Chat Export", title_style))
        simple_story.append(Paragraph(f"Total Chats: {len(chat_history)}", normal_style))
        simple_story.append(Paragraph("Note: Some content could not be exported due to formatting.", small_style))
        
        simple_doc.build(simple_story)
        return simple_filename


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'using_default_key' not in st.session_state:
    st.session_state.using_default_key = False
if 'open_stat_card' not in st.session_state:
    st.session_state.open_stat_card = None
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''
if 'query_timestamp' not in st.session_state:
    st.session_state.query_timestamp = None

def load_models_with_fallback(api_key=None):
    """
    Load vector database and RAG pipeline with fallback handling
    """
    key_to_use = api_key if api_key else DEFAULT_API_KEY
    
    load_start = time.time()
    
    try:
        if not os.path.exists('models/vector_database'):
            st.error("❌ Vector database not found at: models/vector_database/")
            st.info("Make sure your vector database files are in: src/models/vector_database/")
            return None, None, False, False
        
        vector_db = MedicalVectorDB(vector_db_dir='models/vector_database')
        vector_db.load()
        
        pipeline = MedicalRAGPipeline(
            vector_db=vector_db,
            api_key=key_to_use,
            model='gemini-1.5-flash-latest',
            top_k=3,
            similarity_threshold=0.5
        )
        

        load_duration = time.time() - load_start
        model_load_time.observe(load_duration)
        
        st.success(f"✅ System loaded successfully! ({load_duration:.2f}s)")
        return vector_db, pipeline, True, (api_key is None) 
        
    except FileNotFoundError as e:
        st.error("❌ Vector database files not found!")
        st.info("""
        **Please check:**
        1. Files are in `src/models/vector_database/`
        2. You're running from `src/` directory
        3. Files include:
           - faiss_index_20251122_210726.index
           - retrieval_system_20251122_210726.pkl
        """)
        st.code(f"Error details: {str(e)}")
        return None, None, False, False
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if 'api' in error_msg or 'key' in error_msg or 'quota' in error_msg or 'invalid' in error_msg:
            st.error("❌ API Key Error!")
            st.info("""
            **The API key is invalid or has issues.**
            
            Get a free key here: https://makersuite.google.com/app/apikey
            """)
            return None, None, False, False
        else:
            st.error(f"❌ System Error: {str(e)}")
            st.code(f"Full error: {type(e).__name__}: {str(e)}")
            return None, None, False, False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🏥 MED RAG")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        # Clear current results
        st.session_state.current_result = None
        st.session_state.current_query = ''
        st.session_state.query_timestamp = None
        # Clear any open stat cards
        st.session_state.open_stat_card = None
        st.success("✅ Ready for new chat!")
        st.rerun()
    
    st.markdown("---")
    
    if 'pipeline' not in st.session_state or st.session_state.pipeline is None:
        
        if not st.session_state.api_key_set:
            with st.spinner("🔄 Initializing system..."):
                vector_db, pipeline, success, using_default = load_models_with_fallback()
                
                if success:
                    st.session_state.vector_db = vector_db
                    st.session_state.pipeline = pipeline
                    st.session_state.api_key_set = True
                    st.session_state.using_default_key = using_default
                    st.success("✅ System Ready!")
                    st.rerun()
                else:
                    st.session_state.api_key_set = False
                    st.session_state.using_default_key = False
    
    if not st.session_state.api_key_set:
        st.warning("⚠️ Default API key unavailable")
        st.subheader("API Key Required")
        st.info("There's been an issue with the default API key. Please enter a new API key.")
        
        st.markdown("[Get a free API key →](https://makersuite.google.com/app/apikey)")
        
        user_api_key = st.text_input("Your API Key", type="password", key="user_api_input")
        
        if st.button("Use My Key"):
            if user_api_key:
                with st.spinner("🔄 Loading with your key..."):
                    vector_db, pipeline, success, using_default = load_models_with_fallback(user_api_key)
                    
                    if success:
                        st.session_state.vector_db = vector_db
                        st.session_state.pipeline = pipeline
                        st.session_state.api_key_set = True
                        st.session_state.using_default_key = False
                        st.success("✅ System Ready with your key!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key or system error. Please check and try again.")
            else:
                st.error("Please enter an API key")
    else:
        if st.session_state.using_default_key:
            st.success("✅ System Active (Default Key)")
        else:
            st.success("✅ System Active (Your Key)")
        
        st.markdown("---")
        st.subheader("📊 System Info")
        if st.session_state.vector_db:
            st.metric("Documents", f"{len(st.session_state.vector_db.metadata):,}")
            st.metric("Specialties", st.session_state.vector_db.metadata['specialty'].nunique())
        

        st.subheader("💬 Chat History")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):

                actual_index = len(st.session_state.chat_history) - 1 - i
                
                with st.expander(f"{chat['query'][:30]}..."):
                    st.caption(chat['timestamp'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📖 Load", key=f"load_{actual_index}"):
                            st.session_state.current_query = chat['query']
                            st.session_state.current_result = {
                                'answer': chat['answer'],
                                'sources': chat['sources'],
                                'citations': chat['citations']
                            }
                            st.rerun()
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{actual_index}"):
                            st.session_state.chat_history.pop(actual_index)
                            save_chat_history(st.session_state.chat_history)
                            st.success("Deleted!")
                            st.rerun()
            
            st.markdown("---")
            
            if st.button("📥 Export All as PDF"):
                try:
                    pdf_file = export_to_pdf(st.session_state.chat_history)
                    pdf_exports.labels(status='success').inc()
                    
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            f,
                            file_name=os.path.basename(pdf_file),
                            mime="application/pdf"
                        )
                    st.success("✅ PDF generated!")
                except Exception as e:
                    pdf_exports.labels(status='failed').inc()
                    st.error(f"❌ PDF export failed: {str(e)}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                save_chat_history([])
                st.success("✅ History cleared!")
                st.rerun()
        else:
            st.info("No chat history yet")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown("""
<div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; font-size: 3rem; margin: 0;'>🏥 MED RAG System</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;'>
        AI-Powered Medical Knowledge Assistant
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_key_set:
    st.warning("⚠️ Please enter your API key in the sidebar to begin")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📚\n\n5,000+\n\nKnowledge Base", key="stat_kb", use_container_width=True):
        if st.session_state.open_stat_card == 'kb':
            st.session_state.open_stat_card = None
        else:
            st.session_state.open_stat_card = 'kb'
        st.rerun()

with col2:
    if st.button("🏥\n\n8\n\nSpecialties", key="stat_spec", use_container_width=True):
        if st.session_state.open_stat_card == 'spec':
            st.session_state.open_stat_card = None
        else:
            st.session_state.open_stat_card = 'spec'
        st.rerun()

with col3:
    if st.button("🤖\n\nBioBERT\n\nAI Model", key="stat_model", use_container_width=True):
        if st.session_state.open_stat_card == 'model':
            st.session_state.open_stat_card = None
        else:
            st.session_state.open_stat_card = 'model'
        st.rerun()

with col4:
    if st.button("📖\n\nPubMed\n\nCitations", key="stat_pubmed", use_container_width=True):
        if st.session_state.open_stat_card == 'pubmed':
            st.session_state.open_stat_card = None
        else:
            st.session_state.open_stat_card = 'pubmed'
        st.rerun()

if st.session_state.open_stat_card == 'kb':
    st.info("""
    **📚 Knowledge Base**
    
    This system contains over **5,000 peer-reviewed medical research papers** from PubMed, covering diverse medical specialties published between 2020-2024.
    
    Each document includes:
    - Full abstract
    - PubMed ID (PMID)
    - Publication metadata
    - Medical specialty classification
    """)

elif st.session_state.open_stat_card == 'spec':
    st.info("""
    **🏥 Medical Specialties Covered**
    
    1. **Diabetes & Endocrinology**
    2. **Cardiology & Cardiovascular**
    3. **Infectious Diseases**
    4. **Neurology & Brain Disorders**
    5. **Pulmonology & Respiratory**
    6. **Oncology & Cancer Treatment**
    7. **Nephrology & Kidney Disease**
    8. **Gastroenterology & Digestive Health**
    """)

elif st.session_state.open_stat_card == 'model':
    st.info("""
    **🤖 BioBERT Model**
    
    **BioBERT** (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining)
    
    - **Type:** Pre-trained language model specialized for biomedical text
    - **Base Model:** BERT architecture
    - **Training:** Trained on PubMed abstracts and PMC full-text articles
    - **Embedding Dimension:** 768-dimensional vectors
    - **Purpose:** Converts medical text into semantic embeddings for accurate similarity matching
    
    BioBERT outperforms general-purpose models on biomedical NLP tasks by understanding 
    medical terminology, abbreviations, and domain-specific language.
    """)

elif st.session_state.open_stat_card == 'pubmed':
    st.info("""
    **📖 PubMed Citations**
    
    **PubMed** is a free database of biomedical and life sciences literature maintained 
    by the National Center for Biotechnology Information (NCBI).
    
    **Why PubMed?**
    - Over 35 million citations
    - Peer-reviewed research articles
    - Trusted by medical professionals worldwide
    - Regularly updated with latest research
    
    **In this system:**
    - All sources link directly to PubMed
    - Each citation includes PMID for verification
    - Ensures evidence-based, credible information
    """)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("ℹ️ What can I ask about?"):
    st.markdown("""
    ### Supported Medical Specialties:
    - **Diabetes & Endocrinology**
    - **Cardiology & Cardiovascular**
    - **Infectious Diseases**
    - **Neurology & Brain Disorders**
    - **Pulmonology & Respiratory**
    - **Oncology & Cancer Treatment**
    - **Nephrology & Kidney Disease**
    - **Gastroenterology & Digestive Health**
    
    ⚠️ For best results, ask questions within these medical areas.
    """)

st.subheader("🔍 Ask a Medical Question")

query = st.text_area(
    "Enter your question:",
    value=st.session_state.get('current_query', ''),
    height=100,
    placeholder="e.g., What are the symptoms of type 2 diabetes?"
)

if query.strip() and st.session_state.get('open_stat_card'):
    st.session_state.open_stat_card = None
    st.rerun()

with st.expander("💡 Example Questions"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What causes heart disease?"):
            st.session_state.open_stat_card = None
            st.session_state.current_query = "What causes heart disease?"
            st.rerun()
    with col2:
        if st.button("How is stroke treated?"):
            st.session_state.open_stat_card = None
            st.session_state.current_query = "How is stroke treated?"
            st.rerun()
    with col3:
        if st.button("What are COVID-19 symptoms?"):
            st.session_state.open_stat_card = None
            st.session_state.current_query = "What are COVID-19 symptoms?"
            st.rerun()


# ============================================================================
# QUERY PROCESSING
# ============================================================================

if st.button("🔍 Search Medical Literature", type="primary", use_container_width=True):
    st.session_state.open_stat_card = None
    
    if not query.strip():
        st.error("❌ Please enter a question")
    else:
        query_start_timestamp = time.time()
        st.session_state.query_timestamp = query_start_timestamp
        
        with st.spinner("🔍 Searching medical literature..."):
            try:
                result = st.session_state.pipeline.query(query, verbose=False)
                
                duration = time.time() - query_start_timestamp
                query_counter.labels(status='success').inc()
                query_duration.observe(duration)
                
                st.session_state.current_result = result
                st.session_state.current_query = query
                
                max_similarity = 0
                if result['sources']:
                    max_similarity = max(s['similarity_score'] for s in result['sources'])
                    confidence_scores.observe(max_similarity)
                    documents_retrieved.observe(len(result['sources']))
                    
                    for source in result['sources']:
                        specialty_queries.labels(
                            specialty=source.get('specialty', 'Unknown')
                        ).inc()
                
                if max_similarity < 0.7:
                    low_confidence_queries.inc()
                elif max_similarity > 0.85:
                    high_confidence_queries.inc()
                
                if result['citations']:
                    citations_per_response.observe(len(result['citations']))
                
                chat_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'query': query,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'citations': result['citations'],
                    'confidence': max_similarity
                }
                st.session_state.chat_history.append(chat_entry)
                save_chat_history(st.session_state.chat_history)
                
                st.markdown("---")
                
                # ============================================================
                # CONFIDENCE INDICATOR
                # ============================================================
                
                if max_similarity < 0.7:
                    st.warning("""
                    ⚠️ **LIMITED RESULTS FOUND**
                    
                    This query may be outside this specialty coverage. This system specializes in:
                    Diabetes • Cardiology • Infectious Diseases • Neurology • Pulmonology • Oncology • Nephrology • Gastroenterology
                    
                    Results shown below may have limited relevance.
                    """)
                elif max_similarity > 0.85:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                color: white; padding: 1rem; border-radius: 10px; text-align: center;
                                font-weight: 600; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        ✨ HIGH CONFIDENCE RESULTS ✨
                    </div>
                    """, unsafe_allow_html=True)
                
                # ============================================================
                # ANSWER SECTION
                # ============================================================
                
                st.subheader("📝 Answer")
                clean_answer = clean_html_tags(result['answer'])
                
                # Clean up answer text
                clean_answer = clean_answer.replace("Based on medical literature review: ", "")
                clean_answer = clean_answer.replace("Based on medical literature review:", "")
                
                st.markdown(f"""
                <div style='font-size: 20px; line-height: 1.6;'>
                {clean_answer}
                </div>
                """, unsafe_allow_html=True)
                

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"⏱️ Response generated in {duration:.2f}s")
                with col2:
                    if max_similarity > 0:
                        st.caption(f"📊 Confidence: {max_similarity:.2%}")
                
                st.markdown("---")
                
                
                if result['sources']:
                    st.subheader(f"📚 Source Documents ({len(result['sources'])})")
                    
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"[{i}] {source['title'][:80]}..."):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**PMID:** {source['pmid']}")
                                st.markdown(f"**Specialty:** {source['specialty']}")
                                st.markdown(f"**Journal:** {source['journal']}")
                            
                            with col2:
                                similarity_pct = source['similarity_score'] * 100
                                st.metric("Similarity", f"{similarity_pct:.1f}%")
                            
                            st.markdown("**Abstract:**")
                            clean_abstract = clean_html_tags(source['abstract'])
                            st.write(clean_abstract[:300] + "...")
                            
                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"
                            st.markdown(f"[🔗 View on PubMed]({pubmed_url})")
                
                st.markdown("---")
                
                
                if result['citations']:
                    st.subheader("📖 Citations & References")
                    for pmid, cite in result['citations'].items():
                        st.markdown(f"**[PMID: {pmid}]({cite['url']})** - {cite['title'][:60]}...")
                
                
                st.markdown("---")
                
            except Exception as e:
                query_counter.labels(status='error').inc()
                api_errors.labels(error_type=type(e).__name__).inc()
                st.error(f"❌ Error processing query: {str(e)}")

# ============================================================
# USER FEEDBACK SECTION
# ============================================================

from utils.feedback_utils import save_user_feedback
st.markdown("### 🗳️ Feedback")

feedback_col1, feedback_col2 = st.columns([1, 1])

with feedback_col1:
    thumbs_up = st.button("👍", key="thumbs_up_button")

with feedback_col2:
    thumbs_down = st.button("👎", key="thumbs_down_button")

if "show_comment_box" not in st.session_state:
    st.session_state.show_comment_box = False
if "selected_feedback" not in st.session_state:
    st.session_state.selected_feedback = None

if thumbs_up:
    st.session_state.selected_feedback = "up"
    st.session_state.show_comment_box = True
    feedback_counter.labels(feedback_type="positive").inc()

if thumbs_down:
    st.session_state.selected_feedback = "down"
    st.session_state.show_comment_box = True
    feedback_counter.labels(feedback_type="negative").inc()

if st.session_state.show_comment_box:
    st.markdown("#### Optional Comment")
    user_comment = st.text_area("Tell us more (optional)")

    if st.button("Submit Feedback"):
        try:
            save_user_feedback(
                query=st.session_state.current_query,
                answer=st.session_state.current_result['answer'],
                rating="👍" if st.session_state.selected_feedback == "up" else "👎",
                comment=user_comment
            )
            st.success("✅ Thank you! Your feedback has been recorded.")
            st.session_state.show_comment_box = False
        except Exception as e:
            st.error(f"❌ Failed to save feedback: {str(e)}")


# ============================================================================
# PDF EXPORT FOR CURRENT CHAT
# ============================================================================

if 'current_result' in st.session_state and st.session_state.current_result:
    if st.button("📥 Export This Chat as PDF", key="export_current_chat"):
        try:
            export_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': st.session_state.current_query,
                'answer': st.session_state.current_result['answer'],
                'sources': st.session_state.current_result['sources'],
                'citations': st.session_state.current_result['citations']
            }
            
            pdf_file = export_to_pdf([export_entry])
            pdf_exports.labels(status='success').inc()
            
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF",
                    f,
                    file_name=os.path.basename(pdf_file),
                    mime="application/pdf",
                    key="download_current_pdf"
                )
            
            st.success("✅ PDF generated! Click download button above.")
            
        except Exception as e:
            pdf_exports.labels(status='failed').inc()
            st.error(f"❌ PDF export failed: {str(e)}")



st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This system provides information from medical literature for educational purposes only. Always consult healthcare professionals for medical advice.")