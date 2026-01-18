import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import marko

class PNAKnowledgeBase:
    def __init__(self, guide_path):
        self.guide_path = guide_path
        # Force CPU for embeddings to avoid ZeroGPU device mismatch
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.chunks = []
        self.index = None
        
        if os.path.exists(guide_path):
            self._process_guide()
        else:
            print(f"Warning: Guide not found at {guide_path}")

    def _process_guide(self):
        with open(self.guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking by paragraph/section
        # For nursing docs, we want to keep context together
        chunks = content.split('\n\n')
        self.chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
        
        # Create FAISS index
        embeddings = self.encoder.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Knowledge Base initialized with {len(self.chunks)} chunks.")

    def search(self, query, top_k=3):
        if not self.index:
            return ""
        
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = [self.chunks[i] for i in indices[0] if i != -1]
        return "\n\n---\n\n".join(results)

if __name__ == "__main__":
    # Test
    kb = PNAKnowledgeBase("Professional nurse advocate A-EQUIP model Guide.md")
    print(kb.search("What are the four functions of A-EQUIP?"))
