import fitz

doc = fitz.open()
page = doc.new_page()

text = """AI in Education Report

Artificial intelligence is changing how students learn.
Chatbots can answer questions and provide instant feedback.
Teachers can use AI tools to prepare notes, quizzes, and lesson plans.
However, AI should be used responsibly and ethically.
Students must understand concepts instead of only copying AI answers.
"""

page.insert_text((72, 72), text, fontsize=12)

doc.save("day2/sample.pdf")
doc.close()

print("sample.pdf created successfully!")