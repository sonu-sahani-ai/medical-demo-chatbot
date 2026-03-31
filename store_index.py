from dotenv import load_dotenv
import os

from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ==============================
# 🔥 STEP 1: LOAD ENV FILE
# ==============================
load_dotenv(dotenv_path=".env", override=True)

# Debugging
print("Current Path:", os.getcwd())
print("Files:", os.listdir())

# Get API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("PINECONE_API_KEY:", PINECONE_API_KEY)

# ❌ Stop if key missing
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Check your .env file")

# ==============================
# 📄 STEP 2: LOAD DATA
# ==============================
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)

# ==============================
# ✂️ STEP 3: SPLIT TEXT
# ==============================
text_chunks = text_split(filter_data)

print(f"Total Chunks: {len(text_chunks)}")

# ==============================
# 🤖 STEP 4: EMBEDDINGS
# ==============================
embeddings = download_hugging_face_embeddings()

# ==============================
# 🌲 STEP 5: INITIALIZE PINECONE
# ==============================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check existing indexes
existing_indexes = [i.name for i in pc.list_indexes()]
print("Existing Indexes:", existing_indexes)

# Create index if not exists
if index_name not in existing_indexes:
    print("Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # MUST match embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print("Index already exists")

# Connect to index
index = pc.Index(index_name)

# ==============================
# 📦 STEP 6: STORE DATA
# ==============================
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("✅ Data uploaded successfully to Pinecone!")