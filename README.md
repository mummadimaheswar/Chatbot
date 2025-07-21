# Chatbot Deployment (FastAPI + Streamlit)

This project contains a **FastAPI backend** and **Streamlit frontend** that can be deployed easily.

## **Local Deployment (Docker)**
```bash
docker-compose up --build
```
- **API:** http://localhost:8000
- **Frontend:** http://localhost:8501

## **One-Click Deployment on Render**
1. Push this repository to GitHub.
2. Go to [https://render.com](https://render.com).
3. Create a **New Web Service** and connect your GitHub repo.
4. Render detects the `Dockerfile` and deploys your API.
5. You’ll get a live link like `https://chatbot-api.onrender.com`.

---
