from flask import Flask, request, jsonify, render_template
import os
import base64
import io
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key is missing. Please check your .env file.")

genai.configure(api_key=api_key)

# Function to process the PDF using PyMuPDF
def process_pdf(file):
    """Extracts the first page of a PDF as an image and encodes it in Base64."""
    try:
        pdf_bytes = file.read()

        if not pdf_bytes:
            return None, "Uploaded file is empty."

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(doc) == 0:
            return None, "No pages found in the uploaded PDF."

        first_page = doc[0]
        pix = first_page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        return pdf_parts, None

    except Exception as e:
        return None, str(e)

# Function to get response from Gemini API
def get_gemini_response(input_text, pdf_content, prompt):
    """Generate response using Gemini AI (gemini-1.5-flash)."""
    model = genai.GenerativeModel('gemini-1.5-flash')

    input_data = [
        {"role": "user", "parts": [{"text": f"Job Description:\n{input_text}"}]},
        {"role": "user", "parts": pdf_content},
        {"role": "user", "parts": [{"text": f"Instructions:\n{prompt}"}]}
    ]

    try:
        response = model.generate_content(input_data)
        return response.text if response and hasattr(response, "text") else "No response received. Check API inputs."
    except Exception as e:
        return f"API error: {e}"

# Function to format and shorten the output
def format_response(response):
    """Formats the response with Markdown styling and limits it to 200 words."""
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)  # Convert bold text
    response = response.replace("\n", "<br>")  # Convert newlines to HTML breaks

    # Trim to 200 words max
    words = response.split()
    if len(words) > 200:
        response = " ".join(words[:200]) + "..."

    return response

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Endpoint to analyze the resume against job description."""
    job_description = request.form.get("job_description")
    file = request.files.get("resume")

    if not job_description:
        return jsonify({"error": "Missing job description"}), 400
    if not file:
        return jsonify({"error": "No resume uploaded"}), 400

    pdf_content, error = process_pdf(file)
    if error:
        return jsonify({"error": error}), 400

    prompt_analysis = """
    You are an experienced HR professional. Analyze the resume against the job description.
    Highlight the candidate's strengths and weaknesses and suggest improvements.
    Keep the response concise (maximum 200 words).
    """

    response = get_gemini_response(job_description, pdf_content, prompt_analysis)
    formatted_response = format_response(response)

    return render_template("index.html", result=formatted_response)

@app.route('/match', methods=['POST'])
def match_resume():
    """Endpoint to check resume-job match percentage."""
    job_description = request.form.get("job_description")
    file = request.files.get("resume")

    if not job_description:
        return jsonify({"error": "Missing job description"}), 400
    if not file:
        return jsonify({"error": "No resume uploaded"}), 400

    pdf_content, error = process_pdf(file)
    if error:
        return jsonify({"error": error}), 400

    prompt_match = """
    You are an ATS system. Compare the resume with the job description and provide:
    1️⃣ A match percentage.
    2️⃣ Missing keywords.
    3️⃣ Final evaluation.
    Keep the response concise (maximum 200 words).
    """

    response = get_gemini_response(job_description, pdf_content, prompt_match)
    formatted_response = format_response(response)

    return render_template("index.html", result=formatted_response)

if __name__ == '__main__':
    app.run(debug=True)
