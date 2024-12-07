from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """Full preprocessing pipeline for the text."""
        text = self.apply_lemmatization(text)
        return text.strip().lower()

    def apply_lemmatization(self, text):
        """Lemmatize the text."""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(token, self.get_wordnet_pos(tag))
            for token, tag in pos_tags
        ]
        return " ".join(lemmatized)

    def get_wordnet_pos(self, tag):
        """Convert POS tags to WordNet POS tags."""
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def chunk_text(self, text, chunk_size=2048, chunk_overlap=200):
        """Chunk text into smaller segments."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
        )
        return text_splitter.split_text(text)
