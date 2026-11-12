import os
import re
from typing import Dict, List, Any
import pypdf
from bs4 import BeautifulSoup

class BookProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Extract plain text and metadata from .pdf, .txt, or .epub file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("Uploaded file is empty (0 bytes). Please select a valid PDF with text content.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext == ".txt":
            return self._read_txt(file_path)
        elif ext == ".epub":
            return self._read_epub(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload a .pdf or .txt file.")

    def _read_pdf(self, file_path: str) -> Dict[str, Any]:
        text_content = []
        pdf_title = ""
        total_pages = 0

        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
                
                # Try getting title from PDF metadata
                if reader.metadata and reader.metadata.title:
                    pdf_title = str(reader.metadata.title).strip()
                    # Filter out generic titles like "Untitled" or paths
                    if len(pdf_title) < 3 or "/" in pdf_title or "\\" in pdf_title:
                        pdf_title = ""

                for idx, page in enumerate(reader.pages):
                    try:
                        extracted = page.extract_text()
                        if extracted and extracted.strip():
                            text_content.append(extracted.strip())
                    except Exception as pe:
                        continue
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {str(e)}")

        full_text = self.clean_text("\n\n".join(text_content))
        if not full_text.strip() or len(full_text.strip()) < 50:
            raise ValueError("No readable text could be extracted from this PDF. It may contain scanned images or be copy-protected. Please upload a PDF containing searchable text.")

        if not pdf_title:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            pdf_title = base_name.replace("_", " ").replace("-", " ").title()

        return {
            "text": full_text,
            "title": pdf_title,
            "pages": total_pages,
            "word_count": len(full_text.split())
        }

    def _read_txt(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = self.clean_text(f.read())
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        title = base_name.replace("_", " ").replace("-", " ").title()
        return {
            "text": raw_text,
            "title": title,
            "pages": 1,
            "word_count": len(raw_text.split())
        }

    def _read_epub(self, file_path: str) -> Dict[str, Any]:
        import zipfile
        text_content = []
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith(('.html', '.xhtml', '.htm')):
                        with zip_ref.open(name) as html_file:
                            soup = BeautifulSoup(html_file.read(), 'html.parser')
                            text = soup.get_text()
                            if text.strip():
                                text_content.append(text.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read epub file: {e}")

        full_text = "\n\n".join(text_content)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        return {
            "text": full_text,
            "title": base_name.replace("_", " ").replace("-", " ").title(),
            "pages": 1,
            "word_count": len(full_text.split())
        }

    def clean_text(self, text: str) -> str:
        """Normalize line breaks, clean quotes/ligatures, and fix hyphenated PDF breaks."""
        # Standardize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Replace common unicode quotes and hyphens from PDFs
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2014', ' -- ').replace('\u2013', ' - ')
        # Fix line-broken words: e.g. "someth-\ning" -> "something"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Remove repeated excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def split_into_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Split book text by chapter headings or logical structural breaks."""
        pattern = r'(?i)^(chapter\s+[\dIVXLCDM]+|CHAPTER\s+[\dIVXLCDM]+|ACT\s+[\dIVXLCDM]+|BOOK\s+[\dIVXLCDM]+|PROLOGUE|EPILOGUE)'
        lines = text.split('\n')
        chapters = []
        current_title = "Introduction / Opening"
        current_lines = []

        for line in lines:
            if re.match(pattern, line.strip()):
                if current_lines:
                    chapters.append({
                        "title": current_title,
                        "text": "\n".join(current_lines).strip()
                    })
                    current_lines = []
                current_title = line.strip()
            else:
                current_lines.append(line)

        if current_lines:
            chapters.append({
                "title": current_title,
                "text": "\n".join(current_lines).strip()
            })

        return chapters

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk full book text into overlapping semantic passages for RAG embedding."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })
                chunk_id += 1

                # Keep last paragraph for overlap context
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = len("\n\n".join(current_chunk))
            else:
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": "\n\n".join(current_chunk)
            })

        return chunks
