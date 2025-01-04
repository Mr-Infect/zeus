---
# Interactive RAG System with Streamlit

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs, query the content, and receive well-structured answers. If the query is unrelated to the uploaded content, a lightweight language model provides a fallback response. The system is integrated with a **Streamlit GUI** for an intuitive user experience.

---

## Features

- **Resource-Specific Query**: Fetch answers from indexed PDFs.
- **Fallback Mechanism**: Uses a lightweight language model for out-of-scope questions.
- **Partial Matching**: Incorporates fuzzy matching for approximate results.
- **Streamlit Interface**: Simplifies user interactions via a browser-based GUI.
- **Persistent Indexing**: Uses `st.session_state` to manage uploaded resources.

---

## Requirements

Ensure the following dependencies are installed in your environment:

```bash
pip install streamlit faiss-cpu sentence-transformers transformers fuzzywuzzy PyPDF2 torch
```

---

## How It Works

1. **Upload PDFs**:
   - Users upload PDF files via the Streamlit interface.
   - The PDFs are processed, and their content is chunked and indexed using FAISS.

2. **Build FAISS Index**:
   - Create embeddings using `sentence-transformers` and store them in a FAISS index for fast similarity searches.

3. **Ask Questions**:
   - Queries are matched against the indexed content.
   - If a match is found, the system generates a response using a larger language model (e.g., GPT-Neo).
   - For unmatched queries, a fallback lightweight model (e.g., GPT-2) generates the answer.

4. **Switching Mechanism**:
   - Automatically determines whether to use the indexed resources or fallback model without user awareness.

5. **Streamlit Interface**:
   - Easy-to-use GUI with features for uploading files, building the index, and querying.

---

## Commands to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/interactive-rag-system.git
   cd interactive-rag-system
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run zeus.py
   ```

3. Open the URL provided by Streamlit in your browser.

---

## Usage

1. Launch the Streamlit app.
2. Upload one or more PDFs through the interface.
3. Click **"Build FAISS Index"** to process the uploaded files.
4. Enter your query in the text box and hit **"Submit"**.
5. View the answer generated by the system.

---

## Contributing

Feel free to fork the repository, make enhancements, and submit pull requests. Contributions are welcome!

---

## License

This project is licensed under the [MIT License](LICENSE).

---
