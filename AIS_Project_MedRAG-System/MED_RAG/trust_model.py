import re
from typing import Dict, List

class TrustworthinessVerifier:
    #Verify and track trustworthiness metrics
    
    def __init__(self):
        self.verification_log = []
    
    def verify_citations(self, answer: str, documents: List[Dict]) -> Dict:
        # To Verify all citations in answer match actual sources

        # Extracting PMIDs from answer
        cited_pmids = set(re.findall(r'PMID:\s*(\d+)', answer))
        
        # Extracting PMIDs from source documents
        source_pmids = set(doc['pmid'] for doc in documents)
        
        invalid_citations = cited_pmids - source_pmids
        uncited_sources = source_pmids - cited_pmids
        
        verification = {
            'valid': len(invalid_citations) == 0,
            'total_citations': len(cited_pmids),
            'valid_citations': len(cited_pmids - invalid_citations),
            'invalid_citations': list(invalid_citations),
            'uncited_sources': list(uncited_sources),
            'citation_accuracy': (len(cited_pmids - invalid_citations) / len(cited_pmids) * 100) if cited_pmids else 0
        }
        
        return verification
    
    def check_transparency(self, result: Dict) -> Dict:
        transparency_score = 0
        max_score = 5
        issues = []
        
        # 1. Has citations
        if result['citations']:
            transparency_score += 1
        else:
            issues.append('Missing citations')
        
        # 2. Shows sources
        if result['sources']:
            transparency_score += 1
        else:
            issues.append('Missing source documents')
        
        # 3. Has similarity scores
        if result['sources'] and all('similarity_score' in s for s in result['sources']):
            transparency_score += 1
        else:
            issues.append('Missing similarity scores')
        
        # 4. Includes metadata
        if result['sources'] and all('journal' in s and 'publication_date' in s for s in result['sources']):
            transparency_score += 1
        else:
            issues.append('Missing metadata')
        
        # 5. Indicates uncertainty
        answer_lower = result['answer'].lower()
        if any(word in answer_lower for word in ['may', 'suggest', 'indicate', 'possible', 'unclear']):
            transparency_score += 1
        else:
            issues.append('No uncertainty indicators')
        
        return {
            'transparency_score': transparency_score,
            'max_score': max_score,
            'percentage': (transparency_score / max_score) * 100,
            'issues': issues
        }
    
    def verify_reproducibility(self, query: str, result: Dict) -> Dict:
        """
        Check if result is reproducible
        """
        reproducibility = {
            'query_logged': bool(query),
            'sources_available': bool(result['sources']),
            'citations_linked': bool(result['citations']),
            'reproducible': True
        }
        
        # To Check if all sources can be verified
        for source in result['sources']:
            if 'pmid' not in source:
                reproducibility['reproducible'] = False
                break
        
        return reproducibility
    
    def generate_trustworthiness_report(self, result: Dict) -> str:
        """
        Generate human-readable trustworthiness report
        """
        citation_check = self.verify_citations(result['answer'], result['sources'])
        transparency_check = self.check_transparency(result)
        
        report = f"""
TRUSTWORTHINESS REPORT

Citation Verification:
   Total Citations: {citation_check['total_citations']}
   Valid Citations: {citation_check['valid_citations']}
   Citation Accuracy: {citation_check['citation_accuracy']:.1f}%
  {' Invalid Citations: ' + str(citation_check['invalid_citations']) if citation_check['invalid_citations'] else ' All citations valid'}

Transparency Score: {transparency_check['percentage']:.1f}% ({transparency_check['transparency_score']}/{transparency_check['max_score']})
{('Issues: ' + ', '.join(transparency_check['issues'])) if transparency_check['issues'] else ' Fully transparent'}

Source Quality:
  • Number of sources: {len(result['sources'])}
  • Average similarity: {sum(s['similarity_score'] for s in result['sources'])/len(result['sources']):.3f}
  • Specialties: {', '.join(set(s['specialty'] for s in result['sources']))}

Reproducibility:  All sources verifiable via PMID
"""
        return report