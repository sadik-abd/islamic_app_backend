import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from tqdm import tqdm
import json
# Initialize the embedding model
SECRET_KEY = os.environ["GOOGLE_GEMINI_API"] 
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SECRET_KEY)

# Path to root directory containing folders with CSV files
root_dir = "datasets_json"

# Step 2: Extract data from JSON files
def load_data(json_folder):
    documents = []
    metadata_list = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            with open(os.path.join(json_folder, file_name), "r", encoding="utf-8") as file:
                data = json.load(file)
                book_name = data["metadata"]["english"]["title"]
                for hadith in data.get("hadiths", []):
                    english_text = hadith["english"].get("text", "")
                    hadith_number = hadith.get("idInBook", "")
                    
                    # Combine text for embedding
                    combined_text = f"English: {english_text}"
                    documents.append(combined_text)
                    
                    # Add metadata
                    metadata = {
                        "book_name": book_name,
                        "hadith_number": hadith_number,
                    }
                    metadata_list.append(metadata)

    return documents, metadata_list

documents, metadata_list = load_data(root_dir)

# Step 3: Generate embeddings

# Step 4: Create and save the FAISS vector database
vector_db = FAISS.from_texts(documents, embedding_model, metadatas=metadata_list)
vector_db.save_local("faiss_index2")