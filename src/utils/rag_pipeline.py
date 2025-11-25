# rag_pipeline.py

from typing import List, Dict
import os
import re
import google.generativeai as genai


class MedicalRAGPipeline:
    """Complete RAG Pipeline using Google Gemini"""
    
    def __init__(self, vector_db, api_key: str = None, 
                 model: str = 'gemini-1.5-flash-latest',
                 top_k: int = 3,
                 similarity_threshold: float = 0.5,
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        self.model_name = model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_query = 0.0  
        
        self.vector_db = vector_db
        self.llm = None
        
        self.query_count = 0
        self.total_cost = 0.0
        
        self._load_llm()
        print("RAG Pipeline initialized with Gemini")
        
    def _load_llm(self):
        """Load Google Gemini model"""
        genai.configure(api_key=self.api_key)
        
        generation_config = {
            'temperature': self.temperature,
            'max_output_tokens': self.max_tokens,
        }
        
        self.llm = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        print(f"Gemini model loaded: {self.model_name}")
    
    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents"""
        results = self.vector_db.search(query, top_k=self.top_k)
        filtered = [r for r in results if r['similarity_score'] >= self.similarity_threshold]
        return filtered
    
    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not documents:
            return "No relevant medical literature found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context = f"""
[Document {i}]
PMID: {doc['pmid']}
Title: {doc['title']}
Journal: {doc['journal']}
Specialty: {doc['specialty']}

Abstract:
{doc['abstract']}

---"""
            context_parts.append(context)
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini"""
        prompt = f"""You are helping medical researchers by diagnosing based on published literature. Based on the research abstracts below, provide an academic summary for diagnosis.

RESEARCH ABSTRACTS:
{context}

RESEARCH QUESTION: {query}

Provide an evidence-based summary citing studies by PMID number in format [PMID: xxxxx].

ACADEMIC SUMMARY:"""
        return prompt
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini with fallback"""
        academic_prompt = f"""Summarize these medical research findings:

{context}

Question: {query}

Summary with citations [PMID: xxxxx]:"""
    
        try:
            response = self.llm.generate_content(academic_prompt)
            
            if hasattr(response, 'text') and response.text:
                return response.text
            
        except:
            pass
    
    
    # Extract documents from context
        import re
    
    # Find all PMIDs and key content
        pmid_pattern = r'PMID:\s*(\d+)'
        title_pattern = r'Title:\s*(.+?)(?=\n)'
        abstract_pattern = r'Abstract:\s*(.+?)(?=\n---|\Z)'
    
        pmids = re.findall(pmid_pattern, context)
        titles = re.findall(title_pattern, context)
        abstracts = re.findall(abstract_pattern, context, re.DOTALL)
    
        answer_parts = []
    
        answer_parts.append(f"Based on medical literature review:")
    
    # Extracting key information from abstracts
        for i, (pmid, abstract) in enumerate(zip(pmids[:3], abstracts[:3])):
            sentences = abstract.strip().split('. ')[:2]
            summary = '. '.join(sentences) + '.'
            answer_parts.append(f"{summary} [PMID: {pmid}]")
    
        answer_parts.append(f"\n\nThis summary is based on {len(pmids)} research papers. See source documents for complete details.")
    
        return ' '.join(answer_parts)
    
    def extract_citations(self, answer: str, documents: List[Dict]) -> Dict:
        """Extract citations from answer"""
        mentioned_pmids = re.findall(r'PMID:\s*(\d+)', answer)
        citations = {}
    
        for pmid in set(mentioned_pmids):
            # Convert pmid to string for comparison
            pmid_str = str(pmid)
            doc = next((d for d in documents if str(d['pmid']) == pmid_str), None)
            if doc:
                citations[pmid_str] = {
                    'pmid': pmid_str,
                    'title': doc['title'],
                    'journal': doc['journal'],
                    'publication_date': doc['publication_date'],
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
                }
    
        return citations
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        """Main RAG query function"""
        if verbose:
            print(f"Query: {question}")
            
        if verbose:
            print("1. Retrieving documents...")
        documents = self.retrieve_documents(question)
        if verbose:
            print(f"Found {len(documents)} relevant documents")
        
        if not documents:
            return {
                'question': question,
                'answer': "No relevant medical literature found.",
                'citations': {},
                'sources': [],
                'num_sources': 0,
                'cost': 0.0
            }
        
        if verbose:
            print("2. Formatting context...")
        context = self.format_context(documents)
        
        if verbose:
            print("3. Generating answer...")
        answer = self.generate_answer(question, context)
        if verbose:
            print("Answer generated")
        
        if verbose:
            print("4. Extracting citations...")
        citations = self.extract_citations(answer, documents)
        if verbose:
            print(f" {len(citations)} citations found")
        
        self.query_count += 1
        
        if verbose:
            print(f"\nQuery #{self.query_count} ")
        
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'sources': documents,
            'num_sources': len(documents),
            'cost': 0.0
        }
    
    def batch_query(self, questions: List[str], verbose: bool = True) -> List[Dict]:
        """Process multiple questions"""
        results = []
        total = len(questions)
        
        if verbose:
            print(f"BATCH PROCESSING: {total} queries")
        
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[{i}/{total}]")
            result = self.query(question, verbose=verbose)
            results.append(result)
        
        if verbose:
            print("BATCH COMPLETE")
            print(f"Total queries: {total}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        return {
            'total_queries': self.query_count,
            'total_cost': 0.0,
            'avg_cost_per_query': 0.0
        }
    
    def print_result(self, result: Dict):
        #print("RESULT")
        
        print(f"\nQuestion:\n{result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['citations']:
            print(f"\nCitations ({len(result['citations'])}):")
            for pmid, cite in result['citations'].items():
                print(f"\n  [{pmid}] {cite['title']}")
                print(f"  {cite['journal']} ({cite['publication_date']})")
                print(f"  {cite['url']}")
        
        if result['sources']:
            print(f"\nSource Documents ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  {i}. {source['title'][:80]}...")
                print(f"     PMID: {source['pmid']} | Similarity: {source['similarity_score']:.3f}")
        
        print(f"\n printed")