import gradio as gr
import os
import tempfile
from database import store_pdf_content, retrieve_relevant_docs, clear_database
from ollama_chat import generate_response
from model_instructions import get_system_prompt, get_chat_template
from evaluation import ChatbotEvaluator
import PyPDF2
import io
import time

# Initialize session state
chat_history = []
documents_processed = False
evaluator = ChatbotEvaluator()

def process_documents(files):
    """Process uploaded PDF documents."""
    global documents_processed
    if not files:
        return "No files uploaded.", False
    
    try:
        for file in files:
            # Convert Gradio file to bytes
            if hasattr(file, 'name'):
                # For newer Gradio versions
                with open(file.name, 'rb') as f:
                    file_bytes = f.read()
            else:
                # For older Gradio versions
                file_bytes = file
            
            # Process the PDF
            store_pdf_content(file_bytes, os.path.basename(file.name))
        
        documents_processed = True
        return "All documents processed successfully!", True
    except Exception as e:
        return f"Error processing documents: {str(e)}", False

def generate_summary():
    """Generate summary of processed documents."""
    if not documents_processed:
        return "Please upload medical documents first."
    
    try:
        start_time = time.time()
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
        response_time = time.time() - start_time
        
        # Log the interaction
        evaluator.log_interaction(
            query="generate_summary",
            response=summary,
            response_time=response_time
        )
        
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def respond(message, history):
    """Generate response for user questions."""
    global documents_processed
    
    if not documents_processed:
        return history + [[message, "Please upload medical documents first to get personalized responses."]]
    
    try:
        start_time = time.time()
        # Retrieve relevant medical information
        relevant_docs = retrieve_relevant_docs(message)
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

Please provide a concise, precise answer to: {message}

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
        response_time = time.time() - start_time
        
        # Log the interaction
        evaluator.log_interaction(
            query=message,
            response=response,
            response_time=response_time
        )
        
        return history + [[message, response]]
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        evaluator.log_interaction(
            query=message,
            response=error_msg,
            success=False
        )
        return history + [[message, error_msg]]

def get_evaluation_metrics():
    """Get current evaluation metrics."""
    return evaluator.generate_report()

def save_metrics():
    """Save evaluation metrics to file."""
    try:
        evaluator.save_evaluation_data()
        return "Evaluation metrics saved successfully!"
    except Exception as e:
        return f"Error saving metrics: {str(e)}"

def clear_all():
    """Clear all data and reset the application."""
    global documents_processed, chat_history
    try:
        clear_database()
        documents_processed = False
        chat_history = []
        return "All data cleared successfully!", [], "", ""
    except Exception as e:
        return f"Error clearing data: {str(e)}", [], "", ""

# Create the Gradio interface
with gr.Blocks(title="Medical Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Medical Assistant")
    gr.Markdown("Upload medical documents and get instant summaries and answers")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### About")
            gr.Markdown("""
            This medical assistant helps you:
            - Upload and process medical documents
            - Get concise summaries of your medical information
            - Ask questions about your health
            - Receive precise medical guidance
            """)
            
            gr.Markdown("### Instructions")
            gr.Markdown("""
            1. Upload Documents
               Upload your medical PDF documents
            
            2. View Summaries
               Review document summaries
            
            3. Ask Questions
               Ask questions about your health
            
            4. Get Responses
               Receive concise, accurate responses
            """)
            
            clear_btn = gr.Button("Clear All Data")
            
            # Add evaluation metrics section
            gr.Markdown("### Evaluation Metrics")
            metrics_output = gr.Textbox(
                label="Current Metrics",
                lines=15,
                interactive=False
            )
            update_metrics = gr.Button("Update Metrics")
            save_metrics_btn = gr.Button("Save Metrics to File")
        
        with gr.Column(scale=2):
            with gr.Tab("Document Upload"):
                file_output = gr.Textbox(label="Upload Status")
                upload_button = gr.UploadButton(
                    "Upload Medical Documents",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                summary_button = gr.Button("Generate Summary")
                summary_output = gr.Textbox(label="Document Summary", lines=10)
            
            with gr.Tab("Medical Assistant"):
                chatbot = gr.Chatbot(
                    height=400,
                    show_copy_button=True,
                    bubble_full_width=False
                )
                msg = gr.Textbox(
                    label="Ask a medical question",
                    placeholder="e.g., What medications am I currently taking?",
                    show_label=True
                )
                clear = gr.Button("Clear Chat")
    
    # Set up event handlers
    upload_button.upload(
        process_documents,
        inputs=[upload_button],
        outputs=[file_output, gr.State(False)]
    )
    
    summary_button.click(
        generate_summary,
        outputs=summary_output
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",  # This clears the text box after sending
        None,
        msg
    )
    
    clear.click(
        lambda: [],
        None,
        chatbot,
        queue=False
    )
    
    clear_btn.click(
        clear_all,
        outputs=[file_output, chatbot, summary_output, metrics_output]
    )
    
    update_metrics.click(
        get_evaluation_metrics,
        outputs=metrics_output
    )
    
    save_metrics_btn.click(
        save_metrics,
        outputs=file_output
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=False)