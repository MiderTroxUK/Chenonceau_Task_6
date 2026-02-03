
import sys

def extract_text(pdf_path):
    text = ""
    # Try pypdf / PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return "PyPDF2", text
    except ImportError:
        pass
        
    try:
        import pypdf
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return "pypdf", text
    except ImportError:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return "pdfplumber", text
    except ImportError:
        pass
        
    return None, None

files = [
    r"c:\Users\johnh\OneDrive - Cranfield University\Group4 Project\Task3\draftTask3.pdf",
    r"c:\Users\johnh\OneDrive - Cranfield University\Group4 Project\Task3\Pierre.B - Truss - SA.pdf"
]

for file_path in files:
    lib, content = extract_text(file_path)
    if lib:
        print(f"--- EXTRACTED FROM {file_path} USING {lib} ---")
        print(content)
        print("--- END ---")
    else:
        print(f"FAILED TO EXTRACT FROM {file_path}: No suitable library found.")
