#### Summary
**Тематика роботи**- RAG на тематику наукових статей для візуальної одометрії - Відповідальний Максим Орлянський & Френіс Володимир

**Формат даних** - PDF, CSV

**Data source** - зібрані pdf файли із різних наукових журналів  (https://github.com/OrlykM/NLP6) - Відповідальний Максим Орлянський & Френіс Володимир

**Chunking** - langchain, RecursiveCharacterTextSplitter (описаний у файлах preprocessor.py, pdf_data_preprocessor.py) - Відповідальний Френіс Володимир

**LLM** - groq/llama-8b-8192 - Відповідальний Максим Орлянський & Френіс Володимир

**Retriver** - присутні усі (
Sparse BM25 - TF-IDF,  - Відповідальний  Френіс Володимир
Dense - intfloat/e5-large-v2, - Відповідальний  Максим Орлянський & Френіс Володимир
Hybrid-Full - Reranker(BM25+Dense))  Відповідальний Максим Орлянський

**Reranking** - присутній, BAAI/bge-reranker-large - Відповідальний  Френіс Володимир

**Citations** - присутнє - Відповідальний  Френіс Володимир

**UI** - Gradio - Відповідальний Максим Орлянський

**Metadata filtering** - "N/A"

**Source code** - https://github.com/OrlykM/NLP6 - Відповідальний Максим Орлянський

**Задеплоєний сервіс** - .... - Відповідальний Максим Орлянський

** Запитання, де краще BM25 ніж Dense** - 
What is doi number of Comprehensive Performance Evaluation between Visual SLAM and LiDAR SLAM for Mobile Robots: Theories and Experiments ?
> Correct answer: 10.3390/app14093945

**Запитання, де краще Dense ніж BM25** - 
When was released article from reference 90 and what its called for article Visual-SLAM Classical Framework and Key Techniques: A Review  ?
> Correct answer: 2010, Monocular Vision SLAM based on Key Feature Points Selection.
