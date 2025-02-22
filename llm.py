from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import google.generativeai as genai
import time
import soundfile as sf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from scipy.signal import resample
import numpy as np
import torch
import json
import os
SECRET_KEY = os.environ["GOOGLE_GEMINI_API"] 
genai.configure(api_key=SECRET_KEY)
# Define the GeminiModel class
def format_chat_history(chat_history):
    """Converts chat history from the provided format to the Gemini format."""
    formatted_messages = []
    for message in chat_history:
        role = message[0]
        if role == "user":
          role =  "user"
        # For simplicity, assuming anything not "user" is the assistant
        elif role =="bot": #You can expand this logic if you have other roles.
          role ="model"

        formatted_messages.append({"role": role, "parts": message[1]})
    return formatted_messages
class GeminiModel:
    def __init__(self) -> None:
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    def predict_plain(self, inp):
        response = self.model.generate_content(inp)
        
        cost = (response.usage_metadata.total_token_count / 1_000_000) * 10
        txt = response.text.replace('```', '').replace("\n","")
        if "json" in txt[:4]:
            txt = txt[4:]
        return txt, cost

class ShayekhModel:
    def __init__(self) -> None:
        self.prompt = """You are a intelligent chatbot who helps people helps people learn about islam. Or find relevant islamic hadiths or quran ayahs. You will answer their question. you will help them learn more. with your every response you should have a quran ayah or a hadith referenced. use the hadith context if it's related to user question. Context maybe in English. But your response must be in user query's language. if user query is in english please respond in english.
    """
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    def predict_plain(self, inp):
        response = self.model.generate_content(inp)
        
        cost = (response.usage_metadata.total_token_count / 1_000_000) * 10
        txt = response.text.replace('`', '').replace("\n","")
        if "json" in txt[:4]:
            txt = txt[:4]
        return response.text, cost

    def predict(self, inp, history:list, ):
        
        chat  = self.model.start_chat(history=format_chat_history(history))
        chat.send_message
        response = chat.send_message(inp)
        
        cost = (response.usage_metadata.total_token_count / 1_000_000) * 10
        txt = response.text.replace('`', '').replace("\n","")
        if "json" in txt[:4]:
            txt = txt[4:]
        return txt, cost
    
    def generate_query(self, initial_message):
        prompt = f"Based on the user message generate a query that can be used to search a vector db of hadees. Dont Say a extra word. Just return a simple query in english. User:\n\n{initial_message}\n\nQuery:"
        response = self.model.generate_content(prompt)
        title = response.text.strip()
        return title

# Load an existing FAISS database
def load_faiss_db(path_to_db):
    faiss_db = FAISS.load_local(path_to_db, GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SECRET_KEY), allow_dangerous_deserialization=True)
    return faiss_db

def custom_rag_pipeline(query, faiss_db : FAISS, gemini_model : ShayekhModel, history=[]):

    
    search_query = gemini_model.generate_query(query)
    docs = faiss_db.similarity_search(search_query, k=5)
    # Combine retrieved documents into a context string
    contexts = [f"Book name: {doc.metadata['book_name']}\n Hadees Number:{doc.metadata['hadith_number']} \n Text : {doc.page_content}" for doc in docs]
    # try:
    #     context = "\n\n\n\n".join([translator.translate(cont,dest="en",src="ar") for cont in contexts])
    # except Exception as E:
    #     print(E)
    context = "\n\n\n\n".join(contexts)
    
    # Construct a custom prompt
    custom_prompt = f"""
    
    Context:
    {context}
    
    Query: {query}
    Answer:
    """
    
    # Generate response using GeminiModel
    response, cost = gemini_model.predict(custom_prompt,history)
    return {"response":f"{response}","reference":contexts}
def downsample_audio(input_file, target_rate):
    # Read audio file
    data, samplerate = sf.read(input_file)
    
    # Calculate the number of samples after downsampling
    number_of_samples = round(len(data) * float(target_rate) / samplerate)
    
    # Resample the data
    resampled_data = resample(data, number_of_samples)
    
    return resampled_data, target_rate
class SpeechModel:
    def __init__(self) -> None:
        # Load the model with faster-whisper
        self.model = WhisperForConditionalGeneration.from_pretrained("Sadique5/whisper-tiny-quran")
        self.processor = WhisperProcessor.from_pretrained("Sadique5/whisper-tiny-quran")

    def predict(self, inp, txt=""):
        # Process the input audio for the custom model
        print(f"Input Shape {inp['array'].shape}")

        # Average the two channels to get a single channel if stereo
        if len(inp["array"].shape) == 2:
            audio_data = np.mean(inp["array"], axis=1)
        else:
            audio_data = inp["array"]  # If already single channel

        # Convert audio data to tokens for the model
        input_features = self.processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_features
        # Transcribe the audio using the model
        outputs = self.model.generate(input_features)
        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return transcription, []


def process_speech(audio_path, ref_txt):
    whisper_model = SpeechModel()
    mdl = GeminiModel()
    recognized_text,_ = whisper_model.predict({"array":downsample_audio(audio_path,16000)[0]})
    text, money = mdl.predict_plain(f"""we have 2 arabic texts one is reference text and one is text detected by speech recognition. now return data in json format. it should contain following info. your output format should be like this. 
{{"score":how accurate user was this should be a number between 0.0-1.0,"suggestion":suggestion on how user can improve his speech. suggestions should be in english.}} \n\n
Refernce Text: {ref_txt}\nDetection Text {recognized_text}""")
    
    return json.loads(text)



if __name__ == "__main__":
    # model = ShayekhModel()
    # vecdb = load_faiss_db("./faiss_index2")
    # print(custom_rag_pipeline("I want to learn more about jihad",vecdb,model))
    print(process_speech("Recording.mp3","بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"))