
# embedding_utils.py
import pickle
import glob
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_dir='embeddings'):
    """Load the most recent embeddings and metadata"""
    pickle_files = glob.glob(f'{embeddings_dir}/medical_embeddings_complete_*.pkl')

    if not pickle_files:
        raise FileNotFoundError(f"No embedding files found in {embeddings_dir}/")

    latest_file = max(pickle_files, key=os.path.getctime)
    print(f"Loading embeddings from: {latest_file}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ Loaded {data['num_documents']} document embeddings")
    return data

def test_similarity_search(query_text, embeddings_data, model, top_k=5):
    """Test similarity search with a query"""
    query_embedding = model.encode([query_text], normalize_embeddings=True)

    similarities = cosine_similarity(
        query_embedding,
        embeddings_data['embeddings']
    )[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        result = {
            'pmid': embeddings_data['metadata'].iloc[idx]['pmid'],
            'title': embeddings_data['metadata'].iloc[idx]['title'],
            'abstract': embeddings_data['metadata'].iloc[idx]['abstract'][:200] + '...',
            'specialty': embeddings_data['metadata'].iloc[idx]['specialty'],
            'similarity': float(similarities[idx])
        }
        results.append(result)

    return results
