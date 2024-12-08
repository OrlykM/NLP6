import argparse
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

def main(args):
    pdf_path = args.pdf_path
    faiss_index_path = args.faiss_index_path
    use_pdf = args.use_pdf


    pdf_data = DataPdf(pdf_path)
    pdf_embedding = Embedding(model)

    if use_pdf:
        chunk_embeddings = pdf_embedding.generate_embeddings(pdf_data.chunks)
        pdf_embedding.create_faiss_index(chunk_embeddings)
    else:
        pdf_embedding.load_faiss_index(faiss_index_path)

    text_retriever = TextRetriever(
        model,
        pdf_embedding.index,
        pdf_data.chunks,
        pdf_data.chunk_references,
        reranker
    )
    assistant = Assistant(text_retriever, pdf_data.data)

    # Launch UI
    ui = create_ui(assistant.generate_answer)
    ui.launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Assistant with PDF and FAISS index.")
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="./data/odometry_data.csv",
        help="Path to the PDF or CSV data file."
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="./data/faiss_index_e5_large.idx",
        help="Path to the FAISS index file."
    )
    parser.add_argument(
        "--use_pdf",
        action=False,
        help="Generate embeddings from the PDF data instead of loading an existing FAISS index."
    )
    args = parser.parse_args()

    main(args)
