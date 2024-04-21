
# This ChatBot operates without maintaining a state. 
# Each message is stateless and does not retain context from previous messages.

from langchain.prompts import HumanMessagePromptTemplate ,ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

chat = ChatOpenAI()

prompt = ChatPromptTemplate(
    # input_variables = ["content"],
    messages = [
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat , 
    prompt= prompt
)

while True:
    content = input("\n >>: ")
    result = chain({"content":content})
    print(result["text"])