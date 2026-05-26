# 🛒 SmartCart AI: Grocery Store RAG Agent

SmartCart AI is an intelligent, context-aware digital assistant designed for an Asian grocery store. Built using a **Retrieval-Augmented Generation (RAG)** architecture, the agent answers customer queries with high precision by sourcing real-time product data dynamically synced from a Google Sheet.

The assistant is backed by an analytical, detail-oriented persona that strictly adheres to safety guardrails (filtering out toxic language) and remains bound to company-specific product data.

## ✨ Key Features
* **Dynamic Sheet Ingestion:** Seamlessly streams and processes inventory, brands, categories, and pack sizes directly from a Google Sheets CSV export.
* **Semantic Product Search:** Embeds product information using HuggingFace's `all-MiniLM-L6-v2` and indexes it into a high-performance local **FAISS Vector Store**.
* **Autonomous Tool-Calling Agent:** Powered by **LangChain** and **Groq (Llama 3.3)**, the agent autonomously decides when to trigger the `search_products` tool based on natural language user queries.
* **Built-in Guardrails:** Programmed to politely decline inappropriate language and gracefully redirect off-topic questions back to the store's inventory data.

## 🛠️ Tech Stack
* **LLM Orchestration:** LangChain (Agents, Tools, Expression Language)
* **Inference Engine:** Groq Cloud (`llama-3.3-70b-versatile`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace Transformers (`sentence-transformers`)
* **Data Pipelines:** Pandas
