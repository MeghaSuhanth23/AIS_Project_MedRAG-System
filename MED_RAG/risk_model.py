import re
from typing import Dict, List

class RiskManager:
    #Manages safety and risk controls for medical RAG system
    
    def __init__(self):
        # Risk thresholds
        self.min_similarity_threshold = 0.5
        self.min_documents_required = 1
        self.max_answer_length = 1000
        
        # Safety flags
        self.disclaimer_shown = False
        
    def check_query_safety(self, query: str) -> Dict:
        #To check whether query is safe to proceed
        risks = {
            'safe': True,
            'warnings': [],
            'risk_level': 'low'
        }
        
        # Checking for emergency keywords
        emergency_keywords = ['emergency', 'urgent', 'dying', 'overdose', 'suicide', 'severe pain']
        if any(keyword in query.lower() for keyword in emergency_keywords):
            risks['safe'] = False
            risks['warnings'].append('EMERGENCY: This system is not for emergencies. Call 911 or emergency services.')
            risks['risk_level'] = 'critical'
            return risks
        
        # Checking for diagnostic queries (high risk)
        diagnostic_keywords = ['do i have', 'am i', 'diagnose me', 'what is wrong with me']
        if any(keyword in query.lower() for keyword in diagnostic_keywords):
            risks['warnings'].append('WARNING: This system provides information only, not diagnosis.')
            risks['risk_level'] = 'high'
        
        # Checking for treatment queries (medium risk)
        treatment_keywords = ['should i take', 'how much', 'dosage', 'can i stop']
        if any(keyword in query.lower() for keyword in treatment_keywords):
            risks['warnings'].append('CAUTION: Consult healthcare provider for treatment decisions.')
            risks['risk_level'] = 'medium'
        
        return risks
    
    def validate_retrieval(self, documents: List[Dict]) -> Dict:
        #validating retrieval results
        validation = {
            'valid': True,
            'issues': []
        }
        
        # To Check if enough documents retrieved
        if len(documents) < self.min_documents_required:
            validation['valid'] = False
            validation['issues'].append(f'Insufficient sources: {len(documents)} found, need {self.min_documents_required}')
        
        # Checking similarity scores
        low_similarity_docs = [d for d in documents if d['similarity_score'] < self.min_similarity_threshold]
        if low_similarity_docs:
            validation['issues'].append(f'WARNING: {len(low_similarity_docs)} low-confidence sources')
        
        # Checking document currency
        from datetime import datetime
        current_year = datetime.now().year
        old_docs = []
        for doc in documents:
            try:
                pub_year = int(doc['publication_date'][:4])
                if current_year - pub_year > 5:
                    old_docs.append(doc)
            except:
                pass
        
        if old_docs:
            validation['issues'].append(f'INFO: {len(old_docs)} sources older than 5 years')
        
        return validation
    
    def check_answer_safety(self, answer: str, citations: Dict) -> Dict:
        #check generated answer for safety
        safety = {
            'safe': True,
            'flags': []
        }
        
        # Checking for unsupported claims (no citations)
        if not citations:
            safety['flags'].append('RISK: Answer has no citations')
            safety['safe'] = False
        
        # Checking for absolute statements without hedging
        absolute_words = ['always', 'never', 'definitely', 'certainly', 'guaranteed']
        if any(word in answer.lower() for word in absolute_words):
            safety['flags'].append('WARNING: Answer contains absolute statements')
        
        # Checking if answer says "I don't know" or equivalent
        uncertainty_phrases = ['insufficient information', 'not enough', 'cannot determine', 'unclear']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            safety['flags'].append('INFO: Answer indicates uncertainty (GOOD)')
        
        # Checking answer length (too long = might be hallucinating)
        if len(answer) > self.max_answer_length:
            safety['flags'].append('WARNING: Answer unusually long, verify accuracy')
        
        return safety
    
    
    def log_risk_event(self, event_type: str, details: Dict):
        """Log risk events for monitoring"""
        import json
        from datetime import datetime
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        # Append to risk log
        with open('risk_log.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')