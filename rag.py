#%%
try:

    from langchain.document_loaders import PyPDFLoader 
    from langchain_text_splitters import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain_ollama import OllamaEmbeddings
    # from langchain_ollama import OllamaLLM
    from langchain_pinecone import PineconeEmbeddings
    from langchain_pinecone import PineconeVectorStore

    

except Exception as e:
    print(e)


#%%
import os
file_path = r"C:\Users\Spencer\Downloads\Documents\diaspora-directory.PDF"
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")



# %%
#load the document
loader=PyPDFLoader(file_path)
docs=loader.load()
print(docs)
# %%
#chunk the document 
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text=text_splitter.split_documents(docs)
print(text)

# %%
#embeddings
# !pip install langchain_pinecone
from langchain_pinecone import PineconeEmbeddings
pinecone_api_key = "2cd20598-2664-4270-b034-d44106efeed1"
embeddings=PineconeEmbeddings(
    model= "multilingual-e5-large" ,
    api_key=pinecone_api_key
)

# embed=embeddings.embed_query('Hello World')

# print(len(embed))


# %%
#vector store using Pinecone
from langchain_pinecone import PineconeVectorStore
index_name='directory'
deb=PineconeVectorStore(text,embeddings, index_name=index_name)



# %%
#vector store using FAISS
from langchain.vectorstores import FAISS
db=FAISS.from_documents(text,embeddings)
#%%
#retrieval
import logging
try:
    
    logging.basicConfig(level=logging.DEBUG)
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    logging.info(f"Type of retriever: {type(retriever)}")
    retrievers=retriever.invoke(" diaspora directory")
    print(len(retrievers))
    print(retrievers[0].page_content)
except Exception as e:
    logging.error(e)






#%%
# groqllm
# !pip install -qU langchain-groq
from langchain_groq import ChatGroq
groq_api_key='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'
os.environ['GROQ_API_KEY'] = groq_api_key
llm=ChatGroq(model="llama3-8b-8192")

#%%
#prompt template
from langchain.prompts.prompt import PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant agent. Answer the following question based on the given context.
If the answer is not in the context "I don't know."

Context: {context}
Question: {question}
Answer:"""
)
#%%
#qa chain
from langchain.chains import RetrievalQA
qa=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)


#%%
#test
query=' What is diaspora directory '

res1=qa.invoke(query).get('result')
print(f"the rag answer: {res1}")

# res2=llm.invoke(query).content
# print(f"the llm answer: {res2}")
# %%

