from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv


load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model= OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function= embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

retriever = db.as_retriever(search_kwargs={"k": 5})
query = " How much did Microsoft pay to acquire Github"

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)

]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Full result:")
print(result)
print("Content only:")
print(result.content)