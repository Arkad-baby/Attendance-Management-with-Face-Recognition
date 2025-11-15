from dotenv import load_dotenv
from supabase import client,create_client,Client
import os

load_dotenv()

url=os.getenv("url")
API_Key=os.getenv("API_Key")

supabase:Client=create_client(url,API_Key)
