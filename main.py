from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


class TextRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        inputs = tokenizer(request.text, return_tensors="pt",
                           truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        return {
            "input": request.text,
            "predicted_class": int(predicted_class),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
