from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# run ollama on shell
# ollama run phi

# fetch the model
llm = Ollama(model='phi')

# load the documents in the following website
# the documents will be inserted into a vector-store, i.e. word embedding db
loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

# load embeddings for the vector-store
# default is LLAMA2
embeddings = OllamaEmbeddings(model='phi')

# split document and store into vector-db
# FAISS is for Facebook AI Similarity Search
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# define the retriever from vector
retriever = vector.as_retriever()

# define a prompt which uses the recent input and the conversation history
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

# create the chain with the llm, retriever and prompt
# the model will use the retriever to respond the prompt
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# manually create an initial conversation history
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

# ========================================= TEST THE MODEL ====================================

# invoke the chain on the conversation history
#response = retriever_chain.invoke({
#    "chat_history": chat_history,
#    "input": "Tell me how"
#})

# it returns documents about testing in LangSmith
# this is because the LLM generated a new query, combining the chat history with the follow up question
#print(response)
#print()

# ===============================================================================================

# create a new chain to continue the conversation with these retrieved documents in mind
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
# document chain uses the retrieved documents to respond
# it automatically substitutes the relevant docs in {context}
document_chain = create_stuff_documents_chain(llm, prompt)

# create a chain combining the retriever with the generation model
# first get the docs with llm and FAISS according to the prompt
# then, generate the output using the relevant docs
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response['answer'])