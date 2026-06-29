from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import re
import os

app = FastAPI(title="Sentiment Chatbot API")

# Attempt to load the model on startup
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        if os.path.exists('models/sentiment_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
            model = joblib.load('models/sentiment_model.pkl')
            vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

class MessageRequest(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    return text.lower()

@app.post("/api/chat")
def chat_endpoint(request: MessageRequest):
    global model, vectorizer
    if not model or not vectorizer:
        # Retry loading once just in case they were generated after the app started
        load_models()
        if not model or not vectorizer:
            raise HTTPException(status_code=503, detail="Sentiment model is still training or unavailable. Please try again in a moment.")
    
    user_text = request.text
    if not user_text.strip():
        return {"sentiment": "neutral", "confidence": 0, "bot_response": "I didn't quite catch that. Could you say something else?"}
    
    cleaned = clean_text(user_text)
    input_vec = vectorizer.transform([cleaned])
    
    prediction = model.predict(input_vec)[0]
    confidence = model.predict_proba(input_vec)[0].max() * 100
    
    if prediction == 1:
        bot_response = f"**Positive Sentiment Detected!** ({confidence:.1f}% confidence)<br><br>I'm glad to hear that! Keep spreading the good vibes! ✨"
        sentiment_label = "positive"
    else:
        bot_response = f"**Negative Sentiment Detected...** ({confidence:.1f}% confidence)<br><br>I'm sorry to hear that. I hope your day gets better! 💙"
        sentiment_label = "negative"
        
    return {
        "sentiment": sentiment_label,
        "confidence": confidence,
        "bot_response": bot_response
    }

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
