import numpy as np
from rank_bm25 import BM25Okapi


class TextRetriever:
    def __init__(self, model, index, chunks, chunk_references, reranker_model=None):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.chunk_references = chunk_references
        self.reranker = reranker_model  

    def retrieve_with_bm25(self, query, top_n=3):
        tokenized_chunks = [chunk[1].split() for chunk in self.chunks]  
        self.bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query.split()  
        scores = self.bm25.get_scores(tokenized_query)  
        top_chunks_idx = np.argsort(scores)[::-1][:top_n]  
        
        result = {
            "chunks_idx": [self.chunks[i][0] for i in top_chunks_idx], 
            "chunks": [self.chunks[i][1] for i in top_chunks_idx], 
            "scores": [scores[i] for i in top_chunks_idx],  
            "references": [self.chunk_references[i] for i in top_chunks_idx], 
        }
        return result
        
    def retrieve_with_semantic(self, query, top_n=3, rerank=True):
        query_embedding = self.model.encode([query], normalize_embeddings=True)  
        D, I = self.index.search(query_embedding, k=10)  
    
        initial_results = [
            (self.chunk_references[i], self.chunks[i][0], self.chunks[i][1], D[0][j])
            for j, i in enumerate(I[0])
        ]
        if rerank and self.reranker:
            reranker_inputs = [[query, chunk] for _, _, chunk, _ in initial_results]
            reranker_scores = self.reranker.compute_score(reranker_inputs)
            
            reranked_results = sorted(
                zip(initial_results, reranker_scores),
                key=lambda x: x[1],  
                reverse=True         
            )
            
            filtered_results = [
                item for item in reranked_results[:top_n] if item[1] > 0  
            ]
            result = {
                "chunks_idx": [item[0][1] for item in filtered_results],  
                "chunks": [item[0][2] for item in filtered_results],      
                "scores": [item[1] for item in filtered_results],        
                "references": [item[0][0] for item in filtered_results], 
            }

            return result
    
        result = {
            "chunks_idx": [chunk_idx for _, chunk_idx, _ , _ in initial_results[:top_n]],
            "chunks": [chunk for _, _, chunk, _ in initial_results[:top_n]],
            "scores": [dist for _, _, _, dist in initial_results[:top_n]],
            "references": [ref for ref, _, _, _ in initial_results[:top_n]],
        }
        return result

    def retrieve_full_search(self, query, top_n=3, rerank=True):
        semantic_results = self.retrieve_with_semantic(query, top_n)
        bm25_results = self.retrieve_with_bm25(query, top_n)

        bm25_list = self.to_list(bm25_results)
        semantic_list = self.to_list(semantic_results)

        combined_results = bm25_list + semantic_list
        
        seen_chunks = set()
        unique_data = []
        for item in combined_results:
            chunk = item[1]
            if chunk not in seen_chunks:
                unique_data.append(item)  
                seen_chunks.add(chunk) 
        
        a = self.to_dict(unique_data)
        combined_results = list(a['chunk'])  
        
        reranker_inputs = [[query, chunk] for _, chunk, _, _ in unique_data]
        reranker_scores = self.reranker.compute_score(reranker_inputs)
        
        reranked_results = sorted(
            zip(unique_data, reranker_scores),
            key=lambda x: x[1],  
            reverse=True        
        )
        
        filtered_results = [
            item for item in reranked_results[:top_n] if item[1] > -5
        ]

        result = {
            "ranks": list(range(1, top_n + 1)),
            "chunks_idx": [item[0][0] for item in filtered_results],  
            "chunks": [item[0][1] for item in filtered_results],     
            "scores": [item[1] for item in filtered_results],         
            "references": [item[0][2] for item in filtered_results],  
        }

        return result

    def to_list(self, results):
        return list(zip(
            results["chunks_idx"],  
            results["chunks"],      
            results["references"],  
            results["scores"]       
        ))
        
    def to_dict(self, data):
        return {
            "chunk_id": [item[0] for item in data],
            "chunk": [item[1] for item in data],
            "reference": [item[2] for item in data],
            "score": [item[3] for item in data],
        }
