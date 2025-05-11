
# 🩺 Medical Query Bot  
### Intelligent Medical Document Summarizer & Chat Assistant  

**Author:** Madhumitha M  
**Roll Number:** 221501072  
**Department:** AIML-B  

---

## 📌 Project Overview

Medical documents such as lab reports, discharge summaries, or research papers often contain complex terminology and lengthy descriptions, making them hard to understand for non-experts. This project introduces **Medical Query Bot**, a user-friendly system that simplifies and summarizes such documents and enables interactive Q&A using natural language.

The system allows users to:
- Upload medical PDFs  
- Receive simplified, context-aware summaries  
- Ask questions in natural language and get relevant answers  
- Maintain privacy by running completely offline (local system)

---

## ⚙️ Tech Stack

- **Frontend**: Gradio / Streamlit  
- **Backend**: Python  
- **NLP Model**: BERT / LLaMA (via Ollama)  
- **Vector DB**: ChromaDB  
- **Document Processing**: PyPDF2, Sentence Transformers  
- **Evaluation**: ROUGE metrics  

---

## 📈 Performance

- **ROUGE-1 Precision**: 92.3%  
- **F1-Score**: 82.7%  
- **Validation Loss**: 0.214  

These metrics indicate strong summarization and response quality, with minimal information loss.

---

## 🚀 Future Enhancements

- Integration of image-based report analysis (e.g., X-ray or MRI)
- Addition of a mental health support module  
- Multi-language support for regional accessibility  
- Enhanced symptom checker with condition ranking  

---

## 🔧 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-query-bot.git
   cd medical-query-bot
