import fitz  # PyMuPDF
import re
import os

class PDFParser:
    def __init__(self):
        pass

    def clean_text(self, text):
        """Basic cleaning to remove excessive whitespace and common artifacts."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        # Remove hyphenation at the end of lines
        text = re.sub(r'-\n', '', text)
        return text.strip()

    def parse_pdf(self, file_path):
        """Extract text from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at {file_path}")
            
        print(f"Parsing PDF: {os.path.basename(file_path)}")
        text_content = ""
        
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # get_text("blocks") can help preserve paragraph structure better than raw text
                blocks = page.get_text("blocks")
                for b in blocks:
                    block_text = b[4]
                    cleaned_block = self.clean_text(block_text)
                    if cleaned_block:
                        text_content += cleaned_block + "\n\n"
            doc.close()
            return text_content
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""

if __name__ == "__main__":
    # Test
    parser = PDFParser()
    # print(parser.parse_pdf("path/to/test.pdf"))
