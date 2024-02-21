from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# load phi model
llm = Ollama(model='phi')

# define template
template = """Tell me a simple joke about {input}. Return the answer in json format. 
Answer:
"""

# it takes in a dictionary of template variables and produces a PromptValue
# it can be passed either to an LLM or a Chat-based model
prompt = PromptTemplate.from_template(template)

# invoke takes a dictionary as input
# keys are template variables
prompt_value = prompt.invoke({'input': 'ice-cream'})

# show the messages and their type
prompt_messages = prompt_value.to_messages()
print(prompt_messages)

# show the template
prompt_string = prompt_value.to_string()
print(prompt_string)


# the prompt_value can be passed in input to the model
# this line of code is the equivalent of
# chain = prompt | llm
# chain.invoke({'input': 'ice-cream'})
message = llm.invoke(prompt_value)

# the message variable contains model response
# as an AIMessage object
print(message)

# finally, at the end of the chain, there is the Output Parser
# it reads the textual output and converts it into a specific format
# in this case, message is a string representing a Json object
# JsonOutputParser transforms the string into an object
# it is similar to casting
output_parser = JsonOutputParser()
output = output_parser.invoke(message)

print(output)

# every step can be simplified by using LLMChain
# chain = LLMChain(prompt, llm, output_parser)
# chain.invoke({'input': 'ice-cream'})
# LLMChain is equal to
# chain = prompt | llm | output_parser

# ===============================================================================================

""" RAG Example with parallelization """

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser


# load phi embeddings
phi_embeddings = OllamaEmbeddings(model='phi')

# define document vectors in memory
# we’ve setup the retriever using an in memory store, which can retrieve documents based on a query.
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho",
     "bears like to eat honey",
     "luigi has a big house in downtown",
     "marco likes trains",
     "harrison is an amateur chef"],
    embedding=phi_embeddings,
)

# define retriever
retriever = vectorstore.as_retriever()

# define rag template
# the prompt template above takes in context and question as values to be substituted in the prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# define output parser
output_parser = StrOutputParser()

# Before building the prompt template, we want to retrieve relevant documents
# to the search and include them as part of the context.
# RunnableParallel prepare the expected inputs into the prompt
# by using the entries for the retrieved documents as well as the original user question
# RunnablePassthrough simply passes the user question to the next step of the chain
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# define chain
chain = setup_and_retrieval | prompt | llm | output_parser

# execute the chain with the given question
result = chain.invoke({'question': "Where did harrison work?"})

print(result)
