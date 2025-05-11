import ollama

def generate_response(query, context):
    """Generate a response using Ollama and LLaMA 3.2."""
    prompt = f"Use the following information to answer: {context}. Question: {query}"
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
