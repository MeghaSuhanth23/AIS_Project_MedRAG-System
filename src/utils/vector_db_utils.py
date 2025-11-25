# vector_db_utils.py

import faiss
import pickle
import glob
import os
from sentence_transformers import SentenceTransformer

class MedicalVectorDB:
    """Medical Literature Vector Database with FAISS"""

    def __init__(self, vector_db_dir='vector_database'):
        """Initialize and load the vector database"""
        self.vector_db_dir = vector_db_dir
        self.index = None
        self.metadata = None
        self.model = None
        self.model_name = None

    def load(self):
        """Load the most recent vector database"""
        # Find most recent system file
        system_files = glob.glob(f'{self.vector_db_dir}/retrieval_system_*.pkl')

        if not system_files:
            raise FileNotFoundError(f"No retrieval system found in {self.vector_db_dir}/")

        latest_file = max(system_files, key=os.path.getctime)

        print(f"Loading retrieval system from: {latest_file}")

        with open(latest_file, 'rb') as f:
            system = pickle.load(f)

        # FIXED: Build the correct path for index file
        # The system['index_file'] may have old hardcoded path
        # Extract just the filename and combine with current vector_db_dir
        index_filename = os.path.basename(system['index_file'])
        correct_index_path = os.path.join(self.vector_db_dir, index_filename)
        
        print(f"Loading FAISS index from: {correct_index_path}")
        
        # Check if file exists before trying to load
        if not os.path.exists(correct_index_path):
            raise FileNotFoundError(
                f"FAISS index not found at: {correct_index_path}\n"
                f"Make sure {index_filename} is in {self.vector_db_dir}/"
            )
        
        # Load FAISS index with corrected path
        self.index = faiss.read_index(correct_index_path)
        self.metadata = system['metadata']
        self.model_name = system['model_name']

        print(f"✅ Loaded {system['num_documents']} documents")
        print(f"✅ Index type: {system['index_type']}")

        # Load model
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model loaded")

        return self

    def search(self, query, top_k=5):
        """Search for relevant documents"""
        if self.index is None:
            raise ValueError("Database not loaded. Call load() first.")

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            result = {
                'rank': rank,
                'pmid': self.metadata.iloc[idx]['pmid'],
                'title': self.metadata.iloc[idx]['title'],
                'abstract': self.metadata.iloc[idx]['abstract'],
                'specialty': self.metadata.iloc[idx]['specialty'],
                'journal': self.metadata.iloc[idx]['journal'],
                'publication_date': self.metadata.iloc[idx]['publication_date'],
                'similarity_score': float(score)
            }
            results.append(result)

        return results

    def search_by_specialty(self, query, specialty, top_k=5):
        """Search filtered by specialty"""
        # Get more results for filtering
        initial_results = self.search(query, top_k=top_k*3)

        # Filter by specialty
        filtered = [r for r in initial_results if r['specialty'] == specialty]

        return filtered[:top_k]

    def get_document_by_pmid(self, pmid):
        """Retrieve document by PMID"""
        doc = self.metadata[self.metadata['pmid'] == pmid]

        if len(doc) == 0:
            return None

        return doc.iloc[0].to_dict()

    def get_statistics(self):
        """Get database statistics"""
        stats = {
            'total_documents': len(self.metadata),
            'specialties': self.metadata['specialty'].value_counts().to_dict(),
            'model_name': self.model_name,
            'embedding_dimension': self.index.d if self.index else None
        }
        return stats

# # vector_db_utils.py

# import faiss
# import pickle
# import glob
# import os
# from sentence_transformers import SentenceTransformer

# class MedicalVectorDB:
#     """Medical Literature Vector Database with FAISS"""

#     def __init__(self, vector_db_dir='models/vector_database'):
#         """Initialize and load the vector database"""
#         self.vector_db_dir = vector_db_dir
#         self.index = None
#         self.metadata = None
#         self.model = None
#         self.model_name = None

#     def load(self):
#         """Load the most recent vector database"""
#         # Find most recent system file
#         system_files = glob.glob(f'{self.vector_db_dir}/retrieval_system_*.pkl')

#         if not system_files:
#             raise FileNotFoundError(f"No retrieval system found in {self.vector_db_dir}/")

#         latest_file = max(system_files, key=os.path.getctime)

#         print(f"Loading retrieval system from: {latest_file}")

#         with open(latest_file, 'rb') as f:
#             system = pickle.load(f)

#         # Load FAISS index
#         self.index = faiss.read_index(system['index_file'])
#         self.metadata = system['metadata']
#         self.model_name = system['model_name']

#         print(f"Loaded {system['num_documents']} documents")
#         print(f"Index type: {system['index_type']}")

#         # Load model
#         print(f"Loading model: {self.model_name}...")
#         self.model = SentenceTransformer(self.model_name)
#         print("Model loaded")

#         return self

#     def search(self, query, top_k=5):
#         """Search for relevant documents"""
#         if self.index is None:
#             raise ValueError("Database not loaded. Call load() first.")

#         # Generate query embedding
#         query_embedding = self.model.encode([query], normalize_embeddings=True)
#         query_embedding = query_embedding.astype('float32')

#         # Search
#         distances, indices = self.index.search(query_embedding, top_k)

#         # Format results
#         results = []
#         for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
#             result = {
#                 'rank': rank,
#                 'pmid': self.metadata.iloc[idx]['pmid'],
#                 'title': self.metadata.iloc[idx]['title'],
#                 'abstract': self.metadata.iloc[idx]['abstract'],
#                 'specialty': self.metadata.iloc[idx]['specialty'],
#                 'journal': self.metadata.iloc[idx]['journal'],
#                 'publication_date': self.metadata.iloc[idx]['publication_date'],
#                 'similarity_score': float(score)
#             }
#             results.append(result)

#         return results

#     def search_by_specialty(self, query, specialty, top_k=5):
#         """Search filtered by specialty"""
#         # Get more results for filtering
#         initial_results = self.search(query, top_k=top_k*3)

#         # Filter by specialty
#         filtered = [r for r in initial_results if r['specialty'] == specialty]

#         return filtered[:top_k]

#     def get_document_by_pmid(self, pmid):
#         """Retrieve document by PMID"""
#         doc = self.metadata[self.metadata['pmid'] == pmid]

#         if len(doc) == 0:
#             return None

#         return doc.iloc[0].to_dict()

#     def get_statistics(self):
#         """Get database statistics"""
#         stats = {
#             'total_documents': len(self.metadata),
#             'specialties': self.metadata['specialty'].value_counts().to_dict(),
#             'unique_journals': self.metadata['journal'].nunique(),
#             'date_range': f"{self.metadata['publication_date'].min()} to {self.metadata['publication_date'].max()}"
#         }
#         return stats