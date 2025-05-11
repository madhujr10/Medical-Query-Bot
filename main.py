from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import store_pdf_content, retrieve_relevant_docs, clear_database
from ollama_chat import generate_response
from typing import Optional, List
import os
import logging
from fastapi.responses import JSONResponse
from model_instructions import get_system_prompt, get_chat_template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Assistant API",
    description="API for the Medical Assistant using Llama 3.2 with RAG model",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    context_used: bool
    relevant_docs_count: int

DOCUMENTS_FOLDER = "documents"
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        
        clear_database()
        logger.info("Database cleared and initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a medical PDF document and stores its content in ChromaDB."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files (.pdf) are allowed"
            )

        content = await file.read()
        
       
        store_pdf_content(content, file.filename)
        
        logger.info(f"Successfully uploaded and stored file: {file.filename}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Medical document uploaded and processed successfully",
                "filename": file.filename
            }
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing medical document: {str(e)}"
        )

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes medical queries and generates responses using Llama 3.2 with RAG model."""
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

       
        relevant_docs = retrieve_relevant_docs(request.query)
        has_context = bool(relevant_docs)
        context = "\n".join(relevant_docs) if relevant_docs else "No relevant medical information found in the database."

        
        system_prompt = get_system_prompt()
        chat_template = get_chat_template()

    
        prompt = chat_template.format(
            system_prompt=system_prompt,
            user_message=f"""Based on the following medical information:

CONTEXT:
{context}

Please provide a detailed medical response to: {request.query}

Guidelines for your response:
1. If the context contains relevant medical information, use it to provide a comprehensive answer
2. Explain medical terms in simple language
3. Include relevant medical details from the context
4. If the context doesn't contain relevant information, acknowledge this and provide general medical guidance
5. Always maintain a professional medical tone
6. Include appropriate medical disclaimers
7. Suggest consulting healthcare providers when appropriate"""
        )

        response = generate_response(prompt, context)
        
        logger.info(f"Successfully generated medical response for query: {request.query[:50]}...")
        
        return ChatResponse(
            response=response,
            context_used=has_context,
            relevant_docs_count=len(relevant_docs)
        )

    except Exception as e:
        logger.error(f"Error processing medical query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating medical response: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check if the API is running and ready to accept requests."""
    try:
        return {
            "status": "healthy",
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

