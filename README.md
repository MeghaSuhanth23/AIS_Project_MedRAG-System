# AIS_Project_MedRAG-System

An AI-powered Medical Literature Retrieval-Augmented Generation (RAG) system that provides healthcare professionals with evidence-based answers to clinical questions using PubMED medical literature.

## Data Collection Summary

### Dataset Information
- **Source:** PubMED/MEDLINE Database (National Library of Medicine)
- **Total Articles Collected:** 15,212 raw articles
- **Final Dataset:** 14,876 cleaned articles
- **Medical Specialties:** 3 (Cardiology, Diabetes Mellitus, Infectious Diseases)
- **Date Range:** 2020-2024
- **Language:** English only
- **Publication Type:** Peer-reviewed journal articles


## Implementation Components
### 1. Data Collection 
- **API Used:** NCBI E-utilities (Entrez) via Biopython
- **Collection Method:** Automated batch retrieval with rate limiting

### 2. Data Quality Validation 
- **Purpose:** Ensure data completeness and reliability
- **Validation Checks:**
  - Missing critical fields detection
  - Duplicate PMID identification
  - Short abstract filtering (<100 characters)
  - Publication date validation
  - Specialty distribution analysis

### 3. Data Cleaning 
- **Purpose:** Remove noise and standardize data format
- **Cleaning Operations:**
  - Duplicate removal by PMID
  - Missing data handling
  - Text normalization (whitespace, encoding)
  - HTML entity decoding
  - Abstract length recalculation

### 4. Bias Detection 
- **Purpose:** Identify representation biases in medical literature
- **Analysis Dimensions:**
  - Specialty representation balance
  - Temporal distribution analysis
  - Journal diversity assessment
  - Abstract length equity across specialties

### 5. Privacy Compliance
- **Purpose:** Verify no patient-identifiable information collected
- **Privacy Scanning:**
  - Pattern matching for potential identifiers
  - Source verification (public domain)
  - Case report detection
  - Email/contact information scanning

### 6. Data Representativeness Analysis 
- **Purpose:** Ensure diverse medical knowledge coverage
- **Analysis Areas:**
  - Specialty coverage adequacy
  - Temporal range assessment
  - Journal source diversity
  - Content quality distribution


## Data Files

### Raw Data
- `medical_literature_dataset.csv` - Initial collected data (15,212 articles)
- `medical_literature_dataset.json` - JSON backup of raw data

### Cleaned Data
- `medical_literature_cleaned.csv` - After quality filtering (14,876 articles)

### Checkpoint Files
- `data_cardiology_checkpoint.csv` - Cardiology collection checkpoint
- `data_diabetes_checkpoint.csv` - Diabetes collection checkpoint
- `data_infectious_diseases_checkpoint.csv` - Infectious diseases checkpoint

### Reports
- `data_collection_report_[timestamp].txt` - Comprehensive data collection report

## Risk Management Implementation

### Data Collection Risks Addressed
1. **Data Quality Issues**
   - Mitigation: Automated quality validation pipeline
   - Result: 97.8% high-quality data retention

2. **Data Cleaning ad Preprocessing**
   - Clean and preprocess medical data
   - Risk Mitigation: Remove low-quality data and handle inconsistencies

3. **Data Bias**
   - Mitigation: Multi-dimensional bias detection
   - Result: Minimal bias detected (imbalance ratio 1.19)

4. **Privacy Concerns**
   - Mitigation: Automated privacy scanning
   - Result: Zero privacy violations, 100% public domain


## Trustworthiness Implementation

### Strategies Implemented
1. **Fairness:** Statistical bias detection across specialties, temporal, and sources
2. **Transparency:** Complete provenance tracking with PMID verification
3. **Privacy:** Automated privacy compliance verification
4. **Reliability:** Comprehensive quality assurance validation
5. **Accountability:** Detailed documentation and audit trail
