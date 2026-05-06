import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF and extracts all text using PyMuPDF.
    It is written in C and is lightning fast.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text while preserving basic paragraph structure
            full_text += page.get_text("text") + "\n\n"
        return full_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size_words: int = 200, overlap_words: int = 30) -> list[str]:
    """
    Splits the massive judgment into manageable chunks.
    Includes an overlap so no sentences/timelines get cut in half.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    # Step through the text, creating chunks with the specified overlap
    for i in range(0, len(words), chunk_size_words - overlap_words):
        chunk_words = words[i : i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
    return chunks

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Document Processor Initialized.")
    print("To test, put a sample judgment PDF named 'test.pdf' in this folder.")
    
    if os.path.exists("test.pdf"):
        print("Found test.pdf, extracting text...")
        raw_text = extract_text_from_pdf("test.pdf")
        
        my_chunks = chunk_text(raw_text)
        
        print(f"Extraction complete! The document was split into {len(my_chunks)} chunks.")
        if len(my_chunks) > 0:
            print("\nHere is a preview of Chunk 1:")
            print("-" * 50)
            print(my_chunks[0][:200] + "...") # Print first 200 characters