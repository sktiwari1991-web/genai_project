from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv('OPENAI_API_KEY'))

llm = OpenAI(model="gpt-4o-mini")

result = llm.invoke("What is the capital of india")

print(result)