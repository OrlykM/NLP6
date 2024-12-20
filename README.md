## **Задача**  
Розробка Retrieval-Augmented Generation (RAG) системи для роботи з PDF-документами, пов’язаними з науковими статтями на тему візуальної одометрії.

---

## **Опис компонентів**  

- **Data source**  
  PDF-документи (15 штук) зі статтями з наукових журналів. Перед обробкою PDF-документів застосовано лематизацію, щоб звести слова до їх базових форм

- **Chunking**  
  Використано метод RecursiveCharacterTextSplitter з LangChain для розбиття тексту на частини по 2048 символів із перекриттям у 200 символів.  

- **LLM**  
  Використано модель **groq/llama-8b-8192**.  

- **Retriever**  
  Розроблено три методи:  
  1. **BM25** — повертає 3 релевантні документи.  
  2. **Semantic** (на основі **intfloat/e5-large-v2**) — повертає 40 найбільш релевантних документів.  
  3. **FULL search** — комбінує BM25, Semantic + Reranker. Після об’єднання 6 документів (3 від BM25, 3 від Semantic + Reranker), Reranker обирає 3 з них.  

- **Reranker**  
  Використано модель **BAAI/bge-reranker-large**.  

- **Citations**  
  Система повертає назву документа та номер чанка, з якого отримано інформацію.  

- **UI**  
  Інтерфейс реалізовано за допомогою **Gradio**.  

- **Other**  
  **Metadata filtering**: N/A  

---

## **Учасники проєкту**  
- **Максим Орлянський**: Відповідальний за інтеграцію extracting data from pdf, LLM, Hybrid-Full retriever, UI.  
- **Френіс Володимир**: Відповідальний за Sparse BM25, Dense retriever, Reranker, Citations.  

---

## **Посилання на запущений сервіс**  
Сервіс запущений на платформі, де доступний GPU. Без GPU очікування відповідей для Semantic або Full може бути тривалим.  
Для прикладу, сервіс можна запустити на платформі Google Colab: [Посилання на ноутбук](https://colab.research.google.com/drive/1QTObqmkQXHXuOmaWk2QHt4ofHPKboz_D?usp=sharing). 
Для першого запуску, враховуючи встановлення залежностей, потрібно до 10 хвилин. 
