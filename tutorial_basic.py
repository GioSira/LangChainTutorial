import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, XMLOutputParser
from langchain.chains import LLMChain

# setup langsmith
load_dotenv()
os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_API_KEY')


# run ollama on shell
# ollama run phi

# fetch the model
llm = Ollama(model=f'phi')

# ========================================= RUN SIMPLE MODEL ====================================


# run prompt on model via invoke
# res = llm.invoke("how can langsmith help with testing?")

# show output
# print(res)

# ===============================================================================================

# define a prompt template for chat
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are world class technical documentation writer.'),
    ('user', '{input}')
])

# ========================================= RUN CHAT MODEL ====================================

# define a chain --> it takes the prompt template and executes it on the llm
# chain = prompt | llm
# llm_chain = LLMChain(llm=llm, prompt=chat_template)

# invoke the prompt on the model
# res = llm_chain.invoke({'input': 'how can langsmith help with testing?'})

# show chat output
# print(res)

# ===============================================================================================

"""
 simple chat prompt outputs a dict with two keys:
 input: the user prompt
 text: the model output
 
 We can parse text in many ways using output parser
"""

# define a simple text parser
output_parser = StrOutputParser()

# pass the output parser into llm_chain
# chain = prompt | llm | output_parser
llm_chain = LLMChain(llm=llm, prompt=chat_template, output_parser=output_parser)

# invoke the llm_chain on user input
res = llm_chain.invoke({'input': 'how can langsmith help with testing?'})

# show the output
print(res['text'])
