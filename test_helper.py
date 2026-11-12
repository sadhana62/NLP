import os
import asyncio
import httpx
from pypdf import PdfWriter
import io

# Helper to generate a PDF with text
def create_pdf(text_content: str, output_path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text_content.split('\n'):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:90])
        y -= 15
    c.save()

print("Testing PDF creation without reportlab or using simple pypdf...")
