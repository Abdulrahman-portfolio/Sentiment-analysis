from config import API_KEY
import random
import requests


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEAD = {"Authorization": f"Bearer {API_KEY}"}

TH = 0.72

def similarity(q1, q2):
    response = requests.post(
        API,
        headers = HEAD,
        json = {
            "inputs": {
                "source_sentence" : q1, 
                "sentences": [q2]
            }
        }, 
        timeout = 30
    )
    response.raise_for_status()
    return  float(response.json()[0])

def show(q1, q2):
    score = similarity(q2, q2)
    if score >= TH:
        result = "DUPLICATE"
    elif score >= TH -0.05:
        result = "CLOSE MATCH"
    else:
        result = "DIFFERENT"
    print(f"Q1: {q1}")
    print(f"Q1: {q2}")
    print(f"Q1: {q1}")
    print(f"HF Similarity: {score * 100:.1f}%")
    print(f"Result: {result}")

def main():
    print("Hugging face semantic similarity checker")
    print("Type 'exit to quit")

    while True:
        q1 = input("\nSentence 1: ").strip()

        if q1.lower() == "exit":
            break
        q2 = input("Sentence 2: ").strip()

        if q1.lower() == "exit":
            break

        if not q1 or not q2:
            continue

        try:
            show(q1, q2)
        except Exception as e:
            print("Error: ", e)

main()