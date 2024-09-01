import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_TOKEN)

def geminiAPI(user_input):
    response = genai.generate_text(
        prompt=user_input
    )

    # Process the API response
    response_data = response.result
    return json.dumps(response_data, indent=4)