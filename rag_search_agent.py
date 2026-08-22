import os
import pandas as pd
import warnings

# 1. Mute the deprecation warning specifically for langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.tools import StructuredTool
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
# 2. Use the standalone package for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# 3. FAISS stays in community for now (warning is silenced above)
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.Config import Config


tavily_api_key = Config.TAVILY_API_KEY
if not tavily_api_key: 
    raise ValueError("TAVILY_API_KEY is not set in environment or .env")

tavily_client = TavilyClient(api_key=tavily_api_key)


gemini_api_key = Config.GEMINI_API_KEY  # or Config.GOOGLE_API_KEY
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment or .env")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
)

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
    


def search_with_fallback(query: str, k: int = 3) -> str:
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

    # Join the list into a single string for evaluation and return
    combined_text = "\n".join(combined_results)

    if not results or "No relevant documents found." in combined_text:
        print("No relevant documents found in vector store. Searching Tavily.")
        return search_tavily(query)

    print("Local products found. Searching Tavily for additional web information.")
    web_results = search_tavily(query)
    return f"{combined_text}\n\nAdditional Information from web:\n{web_results}"



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

# Use the local search results directly because Groq function-calling is not
# supported reliably in this environment.
search_results = search_with_fallback(user_query, k=3)

prompt = (
    f"{system_prompt}\n\n"
    "Use the information below to answer the user's query.\n\n"
    f"Search results:\n{search_results}\n\n"
    f"Question: {user_query}"
)


def format_response(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)


response_content = llm.invoke(prompt)

print(f"\nUSER: {user_query}")
print(f"AI:\n{format_response(response_content.content)}")
