from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import google.generativeai as genai
import os

# Configure NLTK
nltk.data.path.append("nltk_data") 
# nltk.download('punkt', download_dir="D:/work/newchatbotbuffml_2/nltk_data")

# Configure Google Generative AI
API_KEY = "AIzaSyAcFmNvD4qsT8LSyJzt1mLwg9KtFkHXpq4"
genai.configure(api_key=API_KEY)

# Document Processor Class
class DocumentProcessor:
    def __init__(self):
        self.documents = []  # Stores extracted document texts
        self.tokenized_docs = []  # Tokenized version for BM25

    def extract_text_from_pdf(self, pdf_file_path):
        """Extracts text from a PDF file."""
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    def process_text(self, text):
        """Processes and tokenizes text for retrieval."""
        self.documents.append(text)
        self.tokenized_docs.append(word_tokenize(text.lower()))

    def retrieve_relevant_chunks(self, question, top_n=3):
        """Retrieves the most relevant document chunks for the question."""
        if not self.tokenized_docs:
            return ["No documents available."]
        
        bm25 = BM25Okapi(self.tokenized_docs)
        tokenized_query = word_tokenize(question.lower())
        scores = bm25.get_scores(tokenized_query)

        # Get indices of top N relevant chunks
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [self.documents[i] for i in top_indices]

# Initialize Document Processor
doc_processor = DocumentProcessor()
model = genai.GenerativeModel("gemini-pro")

# Preload PDFs
PDF_FILES = [
    "uploaded_docs/VS_Katze_markiert.pdf",
    "uploaded_docs/VS_Werbung_mit_Tieren.pdf"
]

try:
    for pdf_file_path in PDF_FILES:
        text = doc_processor.extract_text_from_pdf(pdf_file_path)
        print(text)
        if text:
            doc_processor.process_text(text)
        else:
            print(f"Warning: No text extracted from '{pdf_file_path}'.")
    print("All PDFs have been processed successfully.")
except Exception as e:
    print(f"Error processing PDFs: {str(e)}")

# ChatGPT QA System
class ChatGPTQA:
    def __init__(self):
        pass

    def generate_answer(self, context, question):
        """Generates an answer using the AI model."""
        prompt = f"""You are a helpful assistant. Answer the question based on the given context.\n\n
        Context: {context}\n\n
        Question: {question}\n\n
        Answer:"""
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {e}"

# Initialize QA System
qa_system = ChatGPTQA()

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "welcome to dog and cat helps!!"

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles question answering based on processed documents."""
    data = request.get_json()
    question = data.get('question', "").strip()

    if not question:
        return jsonify({"message": "Question cannot be empty."}), 400

    # Retrieve relevant chunks
    try:
        relevant_chunks = doc_processor.retrieve_relevant_chunks(question)
        combined_context = " ".join(relevant_chunks)
    except Exception as e:
        return jsonify({"message": f"Error retrieving relevant chunks: {str(e)}"}), 500

    # Generate answer
    try:
        answer = qa_system.generate_answer(combined_context, question)
        print(f"Relevant Context: {combined_context}")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"message": f"Error generating answer: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
