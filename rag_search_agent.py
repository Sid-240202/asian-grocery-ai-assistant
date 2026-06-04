from langchain_core.tools import StructuredTool
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.Config import Config
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


tavily_api_key = Config.TAVILY_API_KEY
if not tavily_api_key: 
    raise ValueError("TAVILY_API_KEY is not set in environment or .env")

tavily_client = TavilyClient(api_key=tavily_api_key)

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
            "product_name":str(row.get('Product Name (Clean)','')),
            "brand":str(row.get('Brand (Clean)','')),
            "category":str(row.get('Category (Clean)','')),
            "price":str(row.get('Price (Clean)','')),
            "availability":str(row.get('Availability (Clean)',''))
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


def search_tavily(query):
    try:
        print(f"""Fallback triggered. Searching Tavily for: {query}""")
        search_results = tavily_client.search(query = query, search_depth="advanced", max_results=3)

        if not search_results.get('results'):
            return "No results found in Tavily."
        
        web_info = []

        for i, result in enumerate(search_results.get('results', [])[:3],1):
            title = result.get('title', 'Unknown Title - could not fetch')
            content = result.get('content', 'No content available')
            url = result.get('url', 'No URL available')
            web_info.append(f"Source {i}:\nTitle: {title}\nContent: {content}\nURL: {url}\n")

        return f"Information from Tavily:\n\n" + "\n".join(web_info)
        
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return "An error occurred while searching Tavily."
    

def search_with_fallback(query: str, k = 3) -> str:
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 3

    results = vector_store.similarity_search(query, k=k)
    combined_results = []
    for idx, res in enumerate(results):
        product_info = f"====== Product {idx+1} ======\n"
        product_info += res.page_content + "\n"
        combined_results.append(product_info)

    if "No relevant documents found." in combined_results or len(results) == 0:
        print("No relevant documents found in vector store. Falling back to Tavily search.")
        return search_tavily(query)
    
    if len(combined_results) < 50:
        print("Insufficient information from vector store. Falling back to Tavily search.")
        web_results = search_tavily(query)
        return f"{combined_results}\n\nAdditional Information from web:\n{web_results}"
        
    return combined_results



tavily_tool = StructuredTool(
    name = "WebSearch",
    description="Use this tool to search the web for relevant information based on the query.",
    func = search_tavily,
    args_schema=SearchInput
)


search_tool = StructuredTool(
    name="search_products",
    description="Use this tool to search for relevant products based on the user's query.",
    func=search_with_fallback,
    args_schema=SearchInput,
)

system_prompt = (
    "You are a helpful assistant. Use DocumentSearch first, then WebSearch if needed. "
    "Answer the user's question based on internal documents or web search results."
    
)

# For testing, we hardcode the query to avoid blocking input in some environments
user_query = "I want 10 kg atta. which brands are available?"
messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]

# Create the standard required ChatPromptTemplate with placeholders for the executor
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=[search_tool, tavily_tool], prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool, tavily_tool], verbose=True, max_iterations=3)

agent_response = agent_executor.invoke({"input": user_query})
response_content = agent_response["output"]

print(f"\nUSER: {user_query}")
print(f"AI: {response_content}")
