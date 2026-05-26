import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
    SHEET_URL = os.getenv('SHEET_URL', 'https://docs.google.com/spreadsheets/u/1/d/1tdGbkfvcR74sRrppg3fseLOF4pII6CO6/edit?fromCopy=true&ct=2&cct=0')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
