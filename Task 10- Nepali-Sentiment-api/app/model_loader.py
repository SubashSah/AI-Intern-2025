import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device
