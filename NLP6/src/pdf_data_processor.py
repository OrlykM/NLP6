import os
import pandas as pd
from PyPDF2 import PdfReader
import ast
from .preprocessor import TextPreprocessor


class DataPdf:
    def __init__(self, file_path):
        self.data = pd.DataFrame(columns=["name", "text"])
        self.chunk_references = []
        self.chunks = []
        self.preprocessor = TextPreprocessor()

        if file_path.endswith("csv"):
            self.load_data_from_csv(file_path)
        else:
            self.pdf_files = self.get_pdf_files(file_path)
            self.process_pdfs()
            self.preprocess_data()

    def load_data_from_csv(self, file_path):
        """Load data from a CSV file."""
        print("Loading data from CSV...")
        self.data = pd.read_csv(file_path)
        self.data["chunks"] = self.data["chunks"].apply(ast.literal_eval)
        self.extract_chunks_and_references()

    def get_pdf_files(self, directory_path):
        """Get a list of all PDF files in the specified directory."""
        return [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.endswith(".pdf")
        ]

    def process_pdfs(self):
        """Process all PDF files in the given directory."""
        print("Processing PDFs...")
        for pdf_file in self.pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            self.data = pd.concat(
                [
                    self.data,
                    pd.DataFrame(
                        {"name": [os.path.basename(pdf_file)], "text": [text]}
                    ),
                ],
                ignore_index=True,
            )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file."""
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() for page in reader.pages)

    def preprocess_data(self):
        """Preprocess the data: clean text and generate chunks."""
        print("Preprocessing data...")
        self.data["text"] = self.data["text"].apply(self.preprocessor.preprocess_text)
        self.data["chunks"] = self.data["text"].apply(self.preprocessor.chunk_text)
        self.data["indexed_chunks"] = self.data["chunks"].apply(
            lambda chunks: [(i, chunk) for i, chunk in enumerate(chunks)]
        )
        self.extract_chunks_and_references()
        self.save_to_csv("pdf_extracts.csv")

    def extract_chunks_and_references(self):
        """Flatten chunks into a list and associate them with their references."""
        self.chunks.clear()
        self.chunk_references.clear()
        for _, row in self.data.iterrows():
            for i, chunk in enumerate(row["chunks"]):
                self.chunks.append((i, chunk))
                self.chunk_references.append(row["name"])

    def save_to_csv(self, file_path):
        """Save the processed data to a CSV file."""
        self.data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}.")
