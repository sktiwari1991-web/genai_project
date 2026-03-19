from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("Loaded Key:", os.getenv("OPENAI_API_KEY"))

client = OpenAI()

def ask_gpt(question: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    user_question = input("Ask your question: ")
    answer = ask_gpt(user_question)
    print("\nGPT Answer:\n", answer)