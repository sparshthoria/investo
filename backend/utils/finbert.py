from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "yiyanghkust/finbert-tone"

print("Loading FinBERT model...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("✅ FinBERT loaded")


def get_impact_score(text: str) -> float:
    """
    Returns a sentiment impact score between -1.0 and +1.0.
    Score = P(positive) - P(negative)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)

    score = probs[0, 2].item() - probs[0, 0].item()
    return round(score, 4)


def get_sentiment_label(score: float) -> str:
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"