from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header, Depends
from fastapi.responses import JSONResponse
import os
import tempfile
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from llm import *
# FastAPI setup
app = FastAPI()
SECRET_KEY = os.environ.get("SECRET_KEY_APP")
# Models for API requests
class Message(BaseModel):
    query: str

class ChatroomResponse(BaseModel):
    response: str
    reference: List[str]
def verify_secret_key(x_secret_key: str = Header(...)):
    if x_secret_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return "succsex"
# In-memory storage for chatrooms
chatrooms: Dict[str, List[Dict[str, str]]] = {}
shayekh_model = ShayekhModel()
faiss_db = load_faiss_db("faiss_index2")
@app.get("/")
async def homepange(secret_key=Depends(verify_secret_key)):
    return {"message":"app runs fine"}
@app.post("/chatrooms/", response_model=str)
async def create_chatroom(secret_key=Depends(verify_secret_key)):
    chatroom_id = str(uuid4())
    chatrooms[chatroom_id] = [("user",shayekh_model.prompt)]
    return chatroom_id

@app.post("/chatrooms/{chatroom_id}/message", response_model=ChatroomResponse)
async def send_message(chatroom_id: str, message: Message, secret_key=Depends(verify_secret_key)):
    if chatroom_id not in chatrooms:
        raise HTTPException(status_code=404, detail="Chatroom not found")

    history = chatrooms[chatroom_id]
    query = message.query

    try:
        result = custom_rag_pipeline(query=query, faiss_db=faiss_db, gemini_model=shayekh_model, history=history)
        history.append(("user",query))
        history.append(("bot",result['response']))
        return ChatroomResponse(response=result['response'], reference=result['reference'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), ref_txt : str = Form(...), secret_key=Depends(verify_secret_key)):
    """
    Endpoint to transcribe speech from an uploaded audio file.
    
    Args:
        file (UploadFile): The audio file to be transcribed.

    Returns:
        Dict[str, str]: The transcribed text.
    """

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        # Perform inference
        transcription = process_speech(temp_audio_path, ref_txt)
        return transcription

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
