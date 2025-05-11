import streamlit as st
import os
import tempfile
from database import store_pdf_content, retrieve_relevant_docs, clear_database
from ollama_chat import generate_response
from model_instructions import get_system_prompt, get_chat_template
import PyPDF2
import io
import time

# Set page configuration
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="��",
    layout="wide"
)

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.title("Medical Assistant")
st.write("Upload medical documents and get instant summaries and answers")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This medical assistant helps you:
    - Upload and process medical documents
    - Get concise summaries of your medical information
    - Ask questions about your health
    - Receive precise medical guidance
    """)
    
    st.header("Instructions")
    st.write("""
    1. Upload Documents
       Upload your medical PDF documents
    
    2. View Summaries
       Review document summaries
    
    3. Ask Questions
       Ask questions about your health
    
    4. Get Responses
       Receive concise, accurate responses
    """)
    
    if st.button("Clear All Data"):
        with st.spinner("Clearing data..."):
            clear_database()
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.success("All data cleared successfully!")

# Main content
tab1, tab2 = st.tabs(["Document Upload", "Medical Assistant"])

# Document Upload Tab
with tab1:
    st.header("Upload Medical Documents")
    
    st.info("""
    Supported formats: PDF medical documents
    Processing: Documents are processed and stored securely
    Privacy: Your medical information is kept confidential
    """)
    
    uploaded_files = st.file_uploader(
        "Upload your medical PDF documents", 
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.processing = True
            for uploaded_file in uploaded_files:
                # Read the file
                file_bytes = uploaded_file.read()
                
                # Process the PDF
                try:
                    store_pdf_content(file_bytes, uploaded_file.name)
                    st.success(f"Successfully processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            st.session_state.documents_processed = True
            st.session_state.processing = False
            st.success("All documents processed successfully!")
    
    # Document Summarization
    if st.session_state.documents_processed:
        st.header("Document Summaries")
        
        if st.button("Generate Summaries"):
            with st.spinner("Generating summaries..."):
                # Get system prompt
                system_prompt = get_system_prompt()
                chat_template = get_chat_template()
                
                # Create a summary prompt
                summary_prompt = chat_template.format(
                    system_prompt=system_prompt,
                    user_message="""Please provide a concise summary of the medical information in the database. 
                    Focus on:
                    1. Key diagnoses and conditions
                    2. Medications and treatments
                    3. Important medical measurements
                    4. Follow-up requirements
                    
                    Keep the summary brief and to the point."""
                )
                
                # Generate summary
                summary = generate_response(summary_prompt, "")
                
                # Display summary
                st.subheader("Medical Summary")
                st.write(summary)

# Medical Assistant Tab
with tab2:
    st.header("Ask Medical Questions")
    
    if not st.session_state.documents_processed:
        st.warning("Please upload medical documents first to get personalized responses.")
    
    # Chat interface
    user_question = st.text_input("Ask a medical question:", placeholder="e.g., What medications am I currently taking?")
    
    if user_question:
        with st.spinner("Generating response..."):
            # Retrieve relevant medical information
            relevant_docs = retrieve_relevant_docs(user_question)
            has_context = bool(relevant_docs)
            context = "\n".join(relevant_docs) if relevant_docs else ""
            
            # Get the medical chatbot system prompt
            system_prompt = get_system_prompt()
            chat_template = get_chat_template()
            
            # Format the prompt using the chat template
            prompt = chat_template.format(
                system_prompt=system_prompt,
                user_message=f"""Based on the following medical information:

CONTEXT:
{context}

Please provide a concise, precise answer to: {user_question}

Guidelines for your response:
1. Be brief and to the point
2. Focus on the most relevant information
3. If specific information is available, use it
4. If not, provide general medical guidance
5. Always maintain a professional medical tone
6. Include appropriate medical disclaimers when needed"""
            )
            
            # Generate response
            response = generate_response(prompt, context)
            
            # Add to chat history
            st.session_state.chat_history.append({"question": user_question, "answer": response})
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("Conversation History")
        
        for chat in st.session_state.chat_history:
            st.write(f"**Q:** {chat['question']}")
            st.write(f"**A:** {chat['answer']}")
            st.divider()

# Footer
st.markdown("---")
st.caption("Medical Assistant powered by Llama 3.2 | For informational purposes only")
st.caption("Always consult healthcare providers for medical advice") 