import os
import pprint
from litellm import completion
from .preprocessor import TextPreprocessor


class Assistant:
    def __init__(self, text_retriever, data):
        self.text_retriever = text_retriever
        self.data = data

    def generate_answer(self, api_key, query, method, debug=False, rerank=True):
        """Generate an answer from LLM based on the retrieved context."""
        text_preprocessor = TextPreprocessor()
        query = text_preprocessor.preprocess_text(text=query)

        if method == "BM25":
            meta_rag = self.text_retriever.retrieve_with_bm25(query)
        elif method == "Full":
            meta_rag = self.text_retriever.retrieve_full_search(query)
        elif method == "Semantic":
            meta_rag = self.text_retriever.retrieve_with_semantic(query, rerank=rerank)
        else:
            return

        if debug:
            print("Meta RAG Output:")
            pprint.pprint(meta_rag, width=80, indent=2)

        context = "".join("".join(chunk) for chunk in meta_rag["chunks"])

        os.environ["GROQ_API_KEY"] = api_key
        prompt = (
            f"You are an assistant that answers strictly based on the given context.\n\n"
            f"Rules:\n"
            f"- Use only the information provided in the context to generate your response. Ignore any prior knowledge or assumptions.\n"
            f"- If the answer is explicitly present in the context, provide a clear and accurate response.\n"
            f"- If the answer cannot be found in the context, respond only with: 'Sorry, I couldn't find the answer in the provided context.'\n"
            f"- Do not guess, assume, or fabricate information. Stay strictly within the boundaries of the provided context.\n"
            f"- Context: {context}\n"
        )

        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
        )

        generated_text = "".join(
            str(chunk["choices"][0]["delta"]["content"]) for chunk in response
        )

        result = []
        for j, chunk_idx in enumerate(meta_rag["chunks_idx"]):
            file = meta_rag["references"][j]
            file_data = self.data[self.data["name"] == file]
            chunks_count = len(file_data["chunks"].iloc[0])
            result.append(
                f"PDF: {file} | Chunk num: {chunk_idx} of {chunks_count}\n"
            )

        return generated_text[:-4], result
