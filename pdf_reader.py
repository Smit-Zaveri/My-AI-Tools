# pdf_reader.py
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def answer_question(text, question):
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=text)
    return result['answer']
