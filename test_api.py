import os
import google.generativeai as genai
from PIL import Image

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Try getting it from Streamlit secrets
    import toml
    try:
        secrets = toml.load("/Users/Vansh/Desktop/handwritten-document-intelligence/.streamlit/secrets.toml")
        api_key = secrets.get("GEMINI_API_KEY")
    except:
        pass

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
try:
    response = model.generate_content("Hello")
    print("API SUCCESS:", response.text)
except Exception as e:
    print("API ERROR:", str(e))
