import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
text = TextLoader("data/be-good.txt").load()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
vector_db = FAISS.from_documents(text, OpenAIEmbeddings())
retriever = vector_db.as_retriever()

from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)
# output = rag_chain.invoke({"input": "What are the keywords of this article?"})
# print(output["answer"])

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
contextualized_prompt = (
    "Given a chat history and the latest user question and context "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualized_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", contextualized_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

from langchain.chains import create_history_aware_retriever
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt_template)
history_qa_chain = create_stuff_documents_chain(llm, qa_prompt)
history_rag_chain = create_retrieval_chain(aware_retriever, history_qa_chain)

from langchain_core.messages import HumanMessage, AIMessage
chat_history = []

for i in range(0, 6):
    query = str(input("Human : "))
    response = history_rag_chain.invoke({"input": query, "chat_history": chat_history})
    print(response["answer"])
    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=response["answer"]),
        ]
    )





