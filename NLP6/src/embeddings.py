import os
import numpy as np
import faiss
from rich.progress import Progress


class Embedding:
    def __init__(self, model):
        """Initialize the embedding class with a model."""
        self.model = model
        self.index = None

    def generate_embeddings(self, chunks):
        """Generate embeddings for the text chunks."""
        batch_size = 32
        chunk_embeddings = []
        with Progress() as progress:
            task = progress.add_task("Embedding chunks", total=len(chunks))
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    device=self.model.device,
                )
                chunk_embeddings.extend(embeddings)
                progress.update(task, advance=len(batch))
        return np.array(chunk_embeddings)

    def create_faiss_index(self, chunk_embeddings):
        """Create and save a FAISS index."""
        embedding_dim = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  
        self.index.add(chunk_embeddings)
        faiss.write_index(self.index, "faiss_index_e5_large.idx")

    def load_faiss_index(self, filepath):
        """Load an existing FAISS index or initialize a new one."""
        if os.path.exists(filepath):
            print(f"Loading FAISS index from {filepath}")
            self.index = faiss.read_index(filepath)
        else:
            print(f"FAISS index not found at {filepath}")
