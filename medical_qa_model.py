
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import logging

class MedicalQAModel:
    def __init__(self, model_name="ktrapeznikov/biobert_v1.1_pubmed_squad_v2"):
        """
        Initialize the MedicalQAModel with a pretrained model.
        
        :param model_name: Name of the Hugging Face model to load.
        """
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
            logging.info(f"✅ Loaded model: {model_name}")
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise

    def load_context(self, context_text):
        """
        Load a medical report or any text document as context.
        
        :param context_text: Full string of medical data (can be from a PDF or text file).
        """
        if not context_text or not isinstance(context_text, str):
            raise ValueError("Invalid context. Please provide a non-empty string.")
        self.context = context_text.strip()

    def ask_question(self, question):
        """
        Ask a question based on the loaded medical context.
        
        :param question: A string question in natural language.
        :return: Extracted answer string from the context.
        """
        if not hasattr(self, "context"):
            raise RuntimeError("Context not loaded. Call `load_context()` first.")

        if not question or not isinstance(question, str):
            raise ValueError("Invalid question. Must be a non-empty string.")

        result = self.qa_pipeline({
            'context': self.context,
            'question': question
        })

        return result.get('answer', 'No answer found.')

