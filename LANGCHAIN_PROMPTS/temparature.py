from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("Loaded Key:", os.getenv("OPENAI_API_KEY"))

model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

result = model.invoke('write 5 line poem on cricket')

print(result.content)