# FinSolve: Enterprise-Grade RAG Assistant

FinSolve is a production-grade Retrieval-Augmented Generation (RAG) system built to provide secure, grounded, and role-authorized answers from complex enterprise documents.

---

## 🎥 Project Overview
A full demo video is available in the root directory: 
`./Assignment-1_Demo-Sourabh_Deshpande.mp4`

---

## 1. Project Objective
The primary objective of FinSolve is to bridge the gap between unstructured corporate data and actionable employee insights. The system is designed to:
*   **Ensure Groundedness**: Every answer is strictly derived from verified internal documents.
*   **Enforce Security**: A robust Role-Based Access Control (RBAC) system ensures that sensitive department data (e.g., Finance, HR) is only visible to authorized personnel.
*   **Maintain Safety**: Comprehensive input and output guardrails prevent data leakage, PII exposure, and off-topic interactions.

---

## 2. Monorepo Structure
The project is organized into two primary domains:
*   **[backend/](./backend/)**: FastAPI server, RAG logic (Docling/Qdrant), and HR records.
*   **[frontend/](./frontend/)**: React/Next.js client application and Administrative Dashboard.

---

## 3. The User Role and Access Matrix
FinSolve uses a multi-tiered access model to manage data security. Access is determined by a combination of the user's department and their employment status.

| Role Tier | Access Scope | Description |
| :--- | :--- | :--- |
| **C-Level** | **Full Access** | Executives can query documents across all departmental collections. |
| **Departmental** | **Restricted** | Employees (e.g., Marketing, Engineering) can only access their own department's data. |
| **General** | **Public** | All active employees have access to company-wide documents like the Employee Handbook. |
| **Contract/Intern** | **Highly Restricted** | Limited access with custom security messaging for out-of-bounds queries. |

### Assigning User Roles
A list of 500 users is provided in `hr_data.csv`. The active 440 users are assigned roles based on their department and designation. Detailed records are stored in `backend/data/hr/employees.csv`.

### Admins
Designated system administrators have special privileges (e.g., reindexing data, CRUD operations on users). Their details are stored in `backend/data/hr/admins.csv`.

> [!IMPORTANT]
> **Contractors and Interns** receive a specialized security block: *"🚫 Access denied: this information is not allowed to contractual employees and interns."*

---

## 4. Local Setup & Installation

### a. Backend Setup
1. Navigate to the backend directory: `cd backend`
2. Create and activate a virtual environment: `.venv/bin/python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Configure your `.env` file with `GROQ_API_KEY`.
5. Start the server: `.venv/bin/python -m uvicorn app.api:app --reload` (Runs on port 8000).

### b. Frontend Setup
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the application: `npm run dev` (Runs on port 3000).

---

## 5. Technical Architecture

### Document Ingestion (Docling)
*   **Conversion**: PDF and DOCX files are converted to high-fidelity Markdown.
*   **Hierarchical Chunking**: Preserves the relationship between headings and sub-sections.
*   **Metadata Tagging**: Each chunk is tagged with its source and **Required Access Roles**.

### Semantic Routing
Before a query hits the database, it passes through a **Semantic Router** that:
1.  Identifies the intent (Domain Classification).
2.  Acts as an **Access Firewall**, blocking unauthorized cross-departmental queries before search begins.

### Guardrails (Defense in Depth)
*   **Input**: Rate limiting, PII scrubbing, Prompt Injection detection, and Topic filtering.
*   **Output**: Grounding checks (mathematical validation), Role Leakage protection, and Automated Citations.

### Evaluation (RAGAS)
The system undergoes regular **Ablation Studies** to track Faithfulness, Relevancy, and Context Precision on a [0, 1] scale. Results are stored in `backend/evaluation/results/`.

---

## 6. Application Features
*   **Login Screen**: Role-based entry point fetching permissions from the HR database.
*   **Chat Interface**: Contextual, private conversations with verified source citations.
*   **Admin Panel**: Full CRUD operations for employees, document exploration, and one-click database reindexing.

---

*Built with Python (FastAPI/Docling/Qdrant) and React (Next.js/Lucide).*
