
import os
import warnings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool, AgentType

warnings.filterwarnings("ignore")

groq_api_key = os.getenv("GROQ_API_KEY") 


if not groq_api_key:
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()

        
    dotenv_path = os.path.join(base_dir, ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    name, value = line.split("=", 1)
                    if name.strip().upper() == "GROQ_API_KEY":
                        groq_api_key = value.strip().strip('"').strip("'")
                        break


if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")


groq_model = ChatGroq(
    model_name="groq/compound-mini",
    groq_api_key=groq_api_key,
    temperature=0.7,
    max_tokens=512,  # ChatGroq uses max_tokens
)

print("Groq configured successfully!")


documents = [
  "ABC.Informatic was founded in 2010 to deliver innovative IT solutions.",
  "The company specializes in cloud computing, cybersecurity, and enterprise software.",
  "ABC.Informatic has offices in Berlin, Munich, and Hamburg with over 500 employees.",
  "Our mission is to simplify digital transformation for businesses worldwide.",
  "ABC.Informatic invests heavily in research and development to stay ahead in technology."
]
docs = [Document(page_content=doc) for doc in documents]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

def search_documents(query):
    results = vectorstore.similarity_search(query, k=2) #top 2 similar doc
    combined = "\n".join([f"- {res.page_content}" for res in results])
    return f"Relevant Documents:\n{combined}"


search_tool = Tool(
    name="VectorStoreLookup",
    func= search_documents,
    description="Useful for searching relevant information from the document store."
    )

agent = initialize_agent(
    tools=[search_tool],
    llm=groq_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

instructions = (
    "You are a research assistant and an agent. "
    "When asked a question, think step-by-step. "
    "If you need more info, use the VectorStoreLookup tool by writing your thought process, then call the tool. "
    "After getting the info, summarize it and provide a final concise answer to the user. "
    "Do not accept foul language, and do not accept hateful or toxic language. "
    "Do not answer any queries not related to company information. For those, ask the user to ask questions related to company data in the vector DB."
)

query = f"{instructions}\n\nWhat is ABC.Informatic's mission and where are their offices located?"
response = agent.run(query)
print(f"User Query: {query}\n")
print(f"Agent Response: {response}\n")