from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.Config import Config
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

groq_api_key = Config.GROQ_API_KEY

llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile",
               temperature=0.7, max_tokens=1024)

def load_from_google_sheet(sheet_url):
    sheet_id = sheet_url.split('/d/')[1].split('/')[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip()
    duplicates = (df.duplicated()).sum()
    print(f"Found: {duplicates} duplicate rows")


    if duplicates > 0:
        df = df.drop_duplicates(inplace=False)

    print(f"Found Columns: {df.columns.tolist()}\n")

    chunks = []
    for idx, row in df.iterrows():
        if row.isna().all():
            continue

        chunk_parts = []

        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip() != "":
                chunk_parts.append(f"{col}: {value}")
        if not chunk_parts:
            continue

        chunk_text = "\n".join(chunk_parts)


        metadata = {
            "row_number":idx,
            "product_name":str(row.get('Product Name (Clean)','')),
            "brand":str(row.get("Brand",'')),
            "category":str(row.get("Category", '')),
            "pack_size":str(row.get("Pack Size Options",''))
            
            }

        chunks.append({
            "text":chunk_text,
            "metadata":metadata
        })
    return chunks


print(f'='*60)
print("Loading data from Google Sheet.............!!")
print(f'='*60)

sheet_url = Config.SHEET_URL

try: 
    chunks = load_from_google_sheet(sheet_url)
    print(f"Loaded {len(chunks)} chunks from the Google Sheet.")

except Exception as e:
    print(f"Error loading data from Google Sheet: {e}")
    chunks = []
    
docs = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in chunks]

print("Creating embeddings and building vector store.............!!")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding_model)


class SearchInput(BaseModel):
    query: str = Field(description="A natural language query to search for relevant products")


def search_products(query: str, k = 3) -> str:
    """Search for relevant products based on the user's query."""
    results = vector_store.similarity_search(query, k=k)
    
    combined_results = []

    for idx, res in enumerate(results):
        product_info = f"====== Product {idx+1} ======\n"
        product_info += res.page_content + "\n"
        combined_results.append(product_info)
    
    return f"Found {len(results)} relevant products:\n\n" + "\n".join(combined_results)


search_tool = StructuredTool.from_function(
    name="search_products",
    func=search_products,
    args_schema=SearchInput,
    description="Use this tool to search for relevant products based on the user's query. The input should be a natural language query."
)

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt =  """You are a helpful asian grocery store assistant, very experienced, very knowledgeable, analytical and
you do work with detail oriented, professional.

1. When a user asks for a query, use the search_products tool to find relevant products and then provide a
concise answer to the user.
2. Do not accept foul language, do not accept hateful and toxic language.
3. Do not answer any queries not related to company information, for such answers ask user to ask questions related to company or data in vector db.""",
    debug= True
)


user_query = str(input("Type your query here: "))
response = agent.invoke({"messages": [HumanMessage(content=user_query)]})
print(f"\nUSER: {user_query}")
print(f"AI: {response['messages'][-1].content}")
