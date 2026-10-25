from io import BytesIO

from reportlab.pdfgen import canvas


def make_pdf(page_lines: list[list[str]]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for lines in page_lines:
        text = pdf.beginText(50, 790)
        for line in lines:
            text.textLine(line)
        pdf.drawText(text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()
