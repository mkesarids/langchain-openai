# This file demonstrates only File Loading. 

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

# Load the environment variables. In this case it load OpenAI API key if required. 
load_dotenv()

loader = TextLoader('facts.txt')
docs = loader.load()
print(docs)