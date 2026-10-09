import pdfplumber
import docx


def extract_text_from_pdf(pdf_path):
    text = ''

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)

    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)

    return '\n'.join(full_text)