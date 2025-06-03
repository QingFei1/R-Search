import requests

# URL for your local FastAPI server
url = "http://127.0.0.1:8000/retrieve_2wikimultihopqa"
# url = "http://127.0.0.1:8002/retrieve_musique"
# url = "http://127.0.0.1:8001/retrieve_wiki-18"

# Example payload
payload = {
    "queries": ["she moved to England"],
    "topk": 5,
    "return_scores": True
}

# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)
