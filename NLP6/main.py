import nltk
from sentence_transformers import SentenceTransformer
from litellm import completion
from FlagEmbedding import FlagReranker
import gradio as gr
from src.pdf_data_processor import DataPdf
from src.embeddings import Embedding
from src.retriever import TextRetriever
from src.assistant import Assistant
from src.ui import create_ui

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
model = SentenceTransformer('intfloat/e5-large-v2')

print(f"The model is running on: {model.device}")


if __name__ == "__main__":
    use_pdf = False
    pdf_data = DataPdf("./data/odometry_data.csv")

    pdf_embedding = Embedding(model)
    if use_pdf:
        chunk_embeddings = pdf_embedding.generate_embeddings(pdf_data.chunks)
        pdf_embedding.create_faiss_index(chunk_embeddings)
    else:
        pdf_embedding.load_faiss_index("./data/faiss_index_e5_large.idx")

    text_retriever = TextRetriever(
        model,
        pdf_embedding.index,
        pdf_data.chunks,
        pdf_data.chunk_references,
        reranker
    )
    assistant = Assistant(text_retriever, pdf_data.data)

    ui = create_ui(assistant.generate_answer)
    ui.launch(share=True)
