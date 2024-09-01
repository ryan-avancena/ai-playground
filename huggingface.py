import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

def bertLarge(user_input):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    BERT_LARGE_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    def query(payload):
        response = requests.post(BERT_LARGE_URL, headers=headers, json=payload)
        return response.json()
    data = query(
    {
        "inputs": user_input,
        "parameters": {"do_sample": False},
    })
    print(json.dumps(data, indent=4))
    return json.dumps(data, indent=4)


def bertSentiment(user_input):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    BERT_SENTIMENT_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    def query(payload):
        response = requests.post(BERT_SENTIMENT_URL, headers=headers, json=payload)
        return response.json()
    data = query({"inputs": user_input})
    print(json.dumps(data, indent=4))
    return json.dumps(data, indent=4)




# bertSentiment()
# bertLarge()