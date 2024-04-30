# This file demonstrates only File Loading. 

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Load the environment variables. In this case it load OpenAI API key if required. 
load_dotenv()

# Implement TextLoader to split based on Character. 
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,  # Calculate 200 characters,it's gonna try to find the nearest separator character.
    chunk_overlap = 50 # Overlap between documents. This help to avoid cutting words/sentances in half.
)

loader = TextLoader('facts.txt')
docs = loader.load_and_split(text_splitter)

for doc in docs:
    print(doc.page_content + "\n")

# As embeddings calls burn credits. For sample i want to call embeddings only once for one document. 

embeddings = OpenAIEmbeddings()
embedding_result = embeddings.embed_query(docs[0].page_content)
print(embedding_result)