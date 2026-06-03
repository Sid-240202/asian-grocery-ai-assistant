from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.Config import Config
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

groq_api_key = Config.GROQ_API_KEY
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment or .env")

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
            "product_name":str(row.get('Product Name (Clean)',''))
            }

        chunks.append({
            "text":chunk_text,
            "metadata":metadata
        })
    return chunks

print(f'='*60)
print("Loading data from Google Sheet.............!!")
print(f'='*60)

sheet_url = Config.GOOGLE_SHEET_URL
if not sheet_url:
    raise ValueError("GOOGLE_SHEET_URL or SHEET_URL is not set in environment or .env")
try:
    chunks = load_from_google_sheet(sheet_url)
    print(f"Loaded {len(chunks)} chunks from the Google Sheet.")
except Exception as e:
    print(f"Error loading data from Google Sheet: {e}")
    chunks = []

docs = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in chunks]
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
    description="Use this tool to search for relevant products based on the user's query.")

system_prompt = """You are a helpful, highly experienced, analytical, and professional Asian grocery store assistant.

1. Always provide concise and accurate answers to the user based on the context provided.
2. Do not accept foul, hateful, or toxic language.
3. Do not answer any queries not related to company information. For out-of-scope queries, politely ask the user to ask questions related to the grocery store or the product database.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=[search_tool], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# For testing, we hardcode the query to avoid blocking input in some environments
user_query = "I want 10 kg atta. which brands are available?"
response = agent_executor.invoke({"input": user_query})

print(f"\nUSER: {user_query}")
print(f"AI: {response['output']}")