from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
