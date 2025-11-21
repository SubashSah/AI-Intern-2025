import torch
from .config import MAX_LENGTH

def predict_sentiment(text, tokenizer, model, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    pred_id = int(torch.argmax(probs).item())
    score = float(probs[pred_id].item())
    
    # Map id → label cleanly
    id2label = model.config.id2label
    label_name = id2label.get(str(pred_id), id2label.get(pred_id, str(pred_id)))

    return {
        "label_id": pred_id,
        "label": label_name,
        "score": score,
    }
