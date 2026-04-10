# FinSolve Monorepo

Welcome to the FinSolve Enterprise RAG Assistant. The project is organized as a monorepo with a clear separation between the backend logic and the frontend user interface.

## 📂 Project Structure

- **[backend/](./backend/)**: Contains the FastAPI server, RAG pipeline logic (Docling/Qdrant), HR database, and automated evaluation scripts.
- **[frontend/](./frontend/)**: Contains the Next.js web application, chat UI, and administrative dashboard.

---

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
# Create environment and install dependencies
# Add your GROQ_API_KEY to .env
.venv/bin/python -m uvicorn app.api:app --reload
```
Detailed instructions are available in the [Backend README](./backend/README.md).

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Detailed instructions are available in the [Frontend README](./frontend/README.md).

---

## 🎥 Project Overview
A full demo video is available inside the backend directory: 
`./backend/Assignment-1_Demo-Sourabh_Deshpande.mp4`
