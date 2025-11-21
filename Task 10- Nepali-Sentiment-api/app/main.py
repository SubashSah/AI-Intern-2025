from fastapi import FastAPI
from pydantic import BaseModel

from .model_loader import load_model
from .sentiment import predict_sentiment

app = FastAPI(
    title="Nepali Sentiment API",
    description="Sentiment classification using a fine-tuned NepaliBERT model.",
    version="1.0.0",
)

# Load model once
tokenizer, model, device = load_model()

class TextIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Nepali Sentiment API is running."}

@app.post("/predict")
def predict(payload: TextIn):
    return predict_sentiment(payload.text, tokenizer, model, device)
