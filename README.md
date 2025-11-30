# Multimodal AI Chatbot

This project implements a multimodal chatbot capable of processing text and image queries.

## Features
- Text-based chatbot using Hugging Face LLM
- Image understanding using CLIP
- Flask API backend
- Streamlit frontend
- Rate limiting and error handling

## How to Run

1. Clone repository
2. Install dependencies:
   pip install -r requirements.txt
3. Set environment variable:
   HF_TOKEN=your_api_key_here
4. Start backend:
   python backend/app.py
5. Start frontend:
   streamlit run frontend/streamlit_app.py
