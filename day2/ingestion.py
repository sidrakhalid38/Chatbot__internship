import fitz
from docx import Document


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()
    return text


def read_docx(file_path):
    doc = Document(file_path)
    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


if __name__ == "__main__":
    print("TXT FILE CONTENT:")
    print(read_txt("day2/sample.txt"))

    print("\nPDF FILE CONTENT:")
    print(read_pdf("day2/sample.pdf"))

    print("\nDOCX FILE CONTENT:")
    print(read_docx("day2/sample.docx"))