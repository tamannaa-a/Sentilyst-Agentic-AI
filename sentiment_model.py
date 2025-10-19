# sentiment_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import pandas as pd

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

@torch.no_grad()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

@torch.no_grad()
def analyze_sentiment(texts, tokenizer, model):
    labels = ["negative", "neutral", "positive"]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=1).numpy()
    results = []
    for t, p in zip(texts, probs):
        neg, neu, pos = p
        score = float(pos - neg)
        results.append({
            "text": t,
            "negative": round(float(neg), 3),
            "neutral": round(float(neu), 3),
            "positive": round(float(pos), 3),
            "sentiment_score": round(score, 3)
        })
    return pd.DataFrame(results)
