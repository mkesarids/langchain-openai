# This file demonstrates only File Loading. 

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

# Load the environment variables. In this case it load OpenAI API key if required. 
load_dotenv()

embeddings = OpenAIEmbeddings()

emb = embeddings.embed_query("Hello, my name is John Doe. I am a software engineer.")

print(emb)

