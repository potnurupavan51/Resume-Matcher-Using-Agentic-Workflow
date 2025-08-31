# extract_jd_text.py
from docx import Document
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Set up logger
logger = logging.getLogger(__name__)

# PDF extraction imports
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFPageInterpreter
    from io import StringIO
    PDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
        logging.warning("PDF support libraries not found. Only DOCX files will be supported.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a .pdf file using available PDF libraries.

    Args:
        pdf_path (str): The path to the .pdf file.

    Returns:
        str: The concatenated text content from the document.
    """
    if not PDF_SUPPORT:
        raise ImportError("PDF support libraries not installed. Please install pdfminer.six or PyPDF2.")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    
    try:
        # First try pdfminer.six (more reliable for complex PDFs)
        try:
            text = pdf_extract_text(pdf_path)
            return text.strip()
        except:
            # Fallback to PyPDF2
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its extension.
    Supports both .docx and .pdf files.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text content.
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .docx, .pdf")

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extracts all text from a .docx file.

    Args:
        docx_path (str): The path to the .docx file.

    Returns:
        str: The concatenated text content from the document.
    """
    if not os.path.exists(docx_path):
        print(f"Error: DOCX file not found at {docx_path}")
        return ""
    
    try:
        document = Document(docx_path)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return ""

# This __main__ block is for testing this specific module
# if __name__ == "__main__":
#     # Example usage for testing this module independently
#     # Adjust this path if your test file is elsewhere
#     test_file_path = "app/JD_ABAP_Lead.docx" 
#     extracted_text = extract_text_from_docx(test_file_path)

#     if extracted_text:
#         output_txt_file = "test_extracted_jd.txt"
#         with open(output_txt_file, "w", encoding="utf-8") as f:
#             f.write(extracted_text)
#         print(f"Text extracted from '{test_file_path}' and saved to '{output_txt_file}'.")
#         print("\n--- First 500 characters of Extracted Text ---")
#         print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
#     else:
#         print("No text extracted or file not found.")



async def process_single_resume_file_for_ingestion(file_path: Path):
        """Helper to extract text and basic info, handles errors per file. Supports both DOCX and PDF."""
        try:
            # Use the unified text extraction function that handles both DOCX and PDF
            text = extract_text_from_file(str(file_path))
            if not text or not text.strip():
                logger.warning(f"Skipping empty or unreadable resume: {file_path.name}")
                return f"Skipping empty/unreadable resume: {file_path.name}" # Return string error for simplicity

            unique_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_path.name)) # Using filename UUID as primary ID
            name = file_path.stem # Use filename stem as initial name guess

            return {
                "id": unique_id,
                "file_path": str(file_path),
                "filename": file_path.name,
                "raw_text": text,
                "name": name,
                "file_size": file_path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return e # Return the exception object