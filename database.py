import chromadb
from chromadb.config import Settings
import os
import logging
from typing import List
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHROMA_DB_PATH = "chroma_db"
client = chromadb.Client(Settings(
    persist_directory=CHROMA_DB_PATH,
    anonymized_telemetry=False
))


COLLECTION_NAME = "medical_documents"
try:
    collection = client.get_collection(COLLECTION_NAME)
    logger.info(f"Using existing collection: {COLLECTION_NAME}")
except:
    collection = client.create_collection(COLLECTION_NAME)
    logger.info(f"Created new collection: {COLLECTION_NAME}")

DOCUMENTS_FOLDER = "documents"
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def clear_database():
    """Clear all data from the database."""
    try:
        client.delete_collection(COLLECTION_NAME)
        global collection
        collection = client.create_collection(COLLECTION_NAME)
        logger.info("Database cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text content from a PDF file."""
    try:
 
        if not isinstance(pdf_content, bytes):
            logger.error(f"PDF content is not bytes: {type(pdf_content)}")
            raise TypeError(f"PDF content must be bytes, got {type(pdf_content)}")
            
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def store_pdf_content(pdf_content: bytes, filename: str):
    """Store PDF content in the database."""
    try:
 
        if not isinstance(pdf_content, bytes):
            logger.error(f"PDF content is not bytes: {type(pdf_content)}")
            raise TypeError(f"PDF content must be bytes, got {type(pdf_content)}")

        text_content = extract_text_from_pdf(pdf_content)
        
        
        chunks = split_text_into_chunks(text_content)
        
        
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": filename, "chunk": i}],
                ids=[f"{filename}_chunk_{i}"]
            )
        
        logger.info(f"Successfully stored PDF content: {filename}")
    except Exception as e:
        logger.error(f"Error storing PDF content: {str(e)}")
        raise

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def retrieve_relevant_docs(query: str, n_results: int = 5) -> List[str]:
    """Retrieve relevant documents from the database based on the query."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """Splits text into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def store_data_from_markdown(file_path: str):
    """Reads markdown file, chunks text, generates embeddings, and stores in ChromaDB."""
    if not os.path.exists(file_path):
        raise FileNotFoundError("Markdown file not found!")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = chunk_text(content)

    embeddings = embedding_model.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        doc_id = f"{file_path}-{i}"
        collection.add(ids=[doc_id], documents=[chunk], embeddings=[embeddings[i]])

def load_all_markdown_files():
    """Loads all markdown files from the documents folder into ChromaDB."""
    for file_name in os.listdir(DOCUMENTS_FOLDER):
        if file_name.endswith(".md"):
            file_path = os.path.join(DOCUMENTS_FOLDER, file_name)
            store_data_from_markdown(file_path)

