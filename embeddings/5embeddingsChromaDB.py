# This code demonsrates how to create a Chroma DB with OpenAI Embeddings for all the documents which `docs` contains.

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

# Load the environment variables. In this case it load OpenAI API key if required. 
load_dotenv()

# Implement TextLoader to split based on Character. 
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,  # Calculate 200 characters,it's gonna try to find the nearest separator character.
    chunk_overlap = 50 # Overlap between documents. This help to avoid cutting words/sentances in half.
)
# Load the documents from the file.
loader = TextLoader('ap-politics.txt')
docs = loader.load_and_split(text_splitter)

# Below call will create a Chroma DB with OpenAI Embeddings for all the documents which `docs` contains.
db = Chroma.from_documents(docs,
                           embedding=OpenAIEmbeddings(),
                           persist_directory="emb"
                           )

results = db.similarity_search("Jagan Mohan Reddy ")

# When you try to play with k >1 , 
#     we will endup duplicate results as we are embedding the same document multiple times.
#     This will be addressed later. 
# results = db.similarity_search("Jagan Mohan Reddy " , k=5)

for result in results:
    print(result )
    print("\n")
