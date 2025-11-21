# 🧠 Nepali Sentiment Analysis  
Fine-tuned NepaliBERT + FastAPI + Streamlit

This project implements a Nepali-language sentiment classifier using a fine-tuned BERT model.  
The classifier predicts three sentiment labels:

- **0 → Negative**  
- **1 → Neutral**  
- **2 → Positive**

The system includes:

- A **FastAPI backend** for programmatic inference  
- A **Streamlit UI** for interactive testing  
- A fine-tuned model hosted on HuggingFace  
- Clean, modular Python code  

---

## 📂 Project Structure

```project-root/
│
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application
│ ├── config.py # Model constants
│ ├── model_loader.py # Loads tokenizer + model
│ └── sentiment.py # Sentiment prediction logic
│
├── streamlit_app.py # Streamlit UI
├── requirements.txt
└── README.md
```

## 🔗 Fine-Tuned Model

The fine-tuned NepaliBERT model is available on HuggingFace:

👉 **https://huggingface.co/SubashSah/nepali-bert-sentiment**

It is automatically downloaded when FastAPI or Streamlit starts.

---

## ⚙️ 1. Environment Setup

### Create Conda environment (recommended)

```bash
conda create -n nepali-sentiment python=3.11
conda activate nepali-sentiment
```
### Install dependencies
```pip install -r requirements.txt```

## 🚀 2. Running the FastAPI Server

From the project root:

```uvicorn app.main:app --reload```


API will be available at:

http://127.0.0.1:8000

Interactive API docs
http://127.0.0.1:8000/docs

Example Request (POST /predict)
{
  "text": "यो फिल्म एकदमै राम्रो छ।"
}

Example Response
{
  "label_id": 2,
  "label": "positive",
  "score": 0.92
}

## 🎨 3. Running the Streamlit UI

### To launch the user interface:

```streamlit run streamlit_app.py```


The UI opens at:

http://localhost:8501


### Enter Nepali text and get instant output:

```Positive```

```Neutral```

```Negative```

## 🧩 How the System Works

Base model: Shushant/nepaliBERT

Fine-tuned using the Shushant/NepaliSentiment dataset

model_loader.py loads tokenizer & model

sentiment.py performs inference

FastAPI and Streamlit reuse the same prediction logic

## 📦 Dependencies

### Installed via requirements.txt:

fastapi

uvicorn

transformers

torch

streamlit

pydantic

## ✔️ Deliverables Included

Fine-tuned NepaliBERT model (hosted on HuggingFace)

FastAPI backend (/predict endpoint)

Streamlit UI for interactive testing

Modular and readable Python code

Fully reproducible instructions