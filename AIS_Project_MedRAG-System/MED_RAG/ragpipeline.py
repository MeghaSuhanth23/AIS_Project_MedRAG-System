from typing import List, Dict
import os
import re
import google.generativeai as genai
from risk_model import RiskManager
from trust_model import TrustworthinessVerifier


class Medical_RAGPipeline:
    
    def __init__(self, vector_db, api_key: str = None, 
                 model: str = 'gemini-1.5-flash',
                 top_k: int = 3,
                 similarity_threshold: float = 0.5,
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        
        # Configuration
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        self.model_name = model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_query = 0.0 
        
        # Components
        self.vector_db = vector_db
        self.llm = None
        
        # Statistics
        self.query_count = 0
        self.total_cost = 0.0
        
        # Load Gemini
        self._load_llm()
        print(" RAG Pipeline initialized with Gemini")

        self.risk_manager = RiskManager()
        self.trustworthiness_verifier = TrustworthinessVerifier()
        
    def _load_llm(self):
        genai.configure(api_key=self.api_key)
        
        generation_config = {
            'temperature': self.temperature,
            'max_output_tokens': self.max_tokens,
        }
        
        self.llm = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        print(f" Gemini model loaded: {self.model_name}")
    
    def retrieve_documents(self, query: str) -> List[Dict]:
        results = self.vector_db.search(query, top_k=self.top_k)
        filtered = [r for r in results if r['similarity_score'] >= self.similarity_threshold]
        return filtered
    
    def format_context(self, documents: List[Dict]) -> str:
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
        prompt = f"""You are a medical AI assistant helping healthcare professionals. Provide accurate, evidence-based answers with proper citations.

MEDICAL LITERATURE:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided literature
2. Cite sources using [PMID: xxxxx] format
3. If information is insufficient, state this clearly
4. Use professional medical language
5. Highlight any conflicting findings

ANSWER:"""
        return prompt
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini"""
        prompt = self.create_prompt(query, context)
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def extract_citations(self, answer: str, documents: List[Dict]) -> Dict:
        """Extract citations"""
        mentioned_pmids = re.findall(r'PMID:\s*(\d+)', answer)
        citations = {}
        
        for pmid in set(mentioned_pmids):
            doc = next((d for d in documents if d['pmid'] == pmid), None)
            if doc:
                citations[pmid] = {
                    'pmid': pmid,
                    'title': doc['title'],
                    'journal': doc['journal'],
                    'publication_date': doc['publication_date'],
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
        return citations
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        
        risk_assessment = self.risk_manager.check_query_safety(question)

        if not risk_assessment['safe']:
            return {
            'question': question,
            'answer': risk_assessment['warnings'][0],
            'citations': {},
            'sources': [],
            'risk_level': 'critical',
            'safe': False
        }
        
        if risk_assessment['warnings']:
            for warning in risk_assessment['warnings']:
                print(f"\n {warning}\n")
        else:
            print("\n")
            print("-----> NO RISKS <-------")

        if verbose:
            print(f"Query: {question}")
        
        if verbose:
            print("1. Retrieving documents")
        documents = self.retrieve_documents(question)
        if verbose:
            print(f"   Found {len(documents)} relevant documents")
        
        if not documents:
            return {
                'question': question,
                'answer': "No relevant medical literature found.",
                'citations': {},
                'sources': [],
                'num_sources': 0,
                'cost': 0.0
            }
        
        validation = self.risk_manager.validate_retrieval(documents)
        if not validation['valid']:
            print(f"Retrieval validation issues: {validation['issues']}")
        
        if verbose:
            print("2. Formatting context")
        context = self.format_context(documents)
        
        if verbose:
            print("3. Generating answer with Gemini")
        answer = self.generate_answer(question, context)
        if verbose:
            print(" Answer generated")
        
        # Citations
        if verbose:
            print("4. Extracting citations")
        citations = self.extract_citations(answer, documents)
        if verbose:
            print(f"  {len(citations)} citations found")
        
        # Tracking stats
        self.query_count += 1
        
        if verbose:
            print(f"\n Query #{self.query_count} ")
        
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'sources': documents,
            'num_sources': len(documents),
            'cost': 0.0
        }
    
        citation_verification = self.trustworthiness_verifier.verify_citations(answer, documents)
        transparency_check = self.trustworthiness_verifier.check_transparency({
            'answer': answer,
            'citations': citations,
            'sources': documents
        })

        if verbose:
            print(f"   ✓ Citation accuracy: {citation_verification['citation_accuracy']:.1f}%")
            print(f"   ✓ Transparency: {transparency_check['percentage']:.1f}%")

        #Safety Check
        
        answer_safety = self.risk_manager.check_answer_safety(answer, citations)
        if answer_safety['flags']:
            for flag in answer_safety['flags']:
                print(f"   {flag}")
                
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'sources': documents,
            'num_sources': len(documents),
            'cost': 0.0,

            'risk_level': risk_assessment['risk_level'],
            'citation_accuracy': citation_verification['citation_accuracy'],
            'transparency_score': transparency_check['percentage'],
            'safe': answer_safety['safe']
        }
    
    def batch_query(self, questions: List[str], verbose: bool = True) -> List[Dict]:
        results = []
        total = len(questions)
        
        if verbose:
            print(f"BATCH PROCESSING: {total} queries with Gemini")
        
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
        return {
            'total_queries': self.query_count,
            'total_cost': 0.0,
            'avg_cost_per_query': 0.0
        }
    
    def print_result(self, result: Dict):
        
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
        
        print(f"\nDone!!")