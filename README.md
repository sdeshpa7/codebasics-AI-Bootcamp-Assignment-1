# FinSolve: Enterprise-Grade RAG Assistant

FinSolve is a production-grade Retrieval-Augmented Generation (RAG) system built to provide secure, grounded, and role-authorized answers from complex enterprise documents.

A demo video titled Assignment-1_Demo-Sourabh_Deshpande.mp4 is provided in the root directory.

---

## 1. Project Objective
The primary objective of FinSolve is to bridge the gap between unstructured corporate data and actionable employee insights. The system is designed to:
*   **Ensure Groundedness**: Every answer is strictly derived from verified internal documents.
*   **Enforce Security**: A robust Role-Based Access Control (RBAC) system ensures that sensitive department data (e.g., Finance, HR) is only visible to authorized personnel.
*   **Maintain Safety**: Comprehensive input and output guardrails prevent data leakage, PII exposure, and off-topic interactions.

---

## 2. The User Role and Access Matrix
FinSolve uses a multi-tiered access model to manage data security. Access is determined by a combination of the user's department and their employment status.

| Role Tier | Access Scope | Description |
| :--- | :--- | :--- |
| **C-Level** | **Full Access** | Executives can query documents across all departmental collections. |
| **Departmental** | **Restricted** | Employees (e.g., Marketing, Engineering) can only access their own department's data. |
| **General** | **Public** | All active employees have access to company-wide documents like the Employee Handbook. |
| **Contract/Intern** | **Highly Restricted** | Limited access with custom security messaging for out-of-bounds queries. |

### User and Assigning Them User Roles

A list of 500 users is provided in the hr_data.csv file of which 440 users are active with the remaining on notice, resigned or terminated. The active users are assigned roles based on their department and designation. A detailed list can be found in data/hr/employees.csv file with User Roles under the access_role column.

For e.g., All employees within the HR department are provided with the access role of HR and all employees within the Finance department are provided with the access role of Finance. Similarly, all employees within the Data and Technology department are provided with the access role of Engineering and all employees within the Marketing department are provided with the access role of Marketing. C-Level access role is provided to all Full-time active employees with designation VP, irrespetive of department.

### Admins

5 employees are provided with the access role of Admin. There details can be found in the data/hr/employees.csv file.




> [!IMPORTANT]
> **Contractors and Interns** receive a specialized security block: *"🚫 Access denied: this information is not allowed to contractual employees and interns."*

---

## 3. The Data Sources
The system ingests and processes data from diverse enterprise sources:
*   **HR Records**: A structured `data/hr/employees.csv` containing IDs, roles, departments, and employment types.
*   **Departmental Documents**:
    *   **Finance**: Quarterly reports, budget forecasts (PDF/DOCX).
    *   **Marketing**: Campaign strategies, ROI reports (PDF/DOCX).
    *   **Engineering**: Technical specs, architecture diagrams (Markdown/PDF).
    *   **HR**: Policy documents, onboarding guides (PDF/DOCX).
    *   **General**: The global Employee Handbook.

---

## 4. Document Ingestion with Hierarchical Chunking
To maintain the structural integrity of complex documents, FinSolve utilizes the **Docling** library.

*   **Conversion**: PDF and DOCX files are converted to high-fidelity Markdown.
*   **Hierarchical Chunking**: Using `HierarchicalChunker`, the system preserves the relationship between headings and sub-sections.
*   **Metadata Enrichment**: Each chunk is tagged with its source filename, page number, and **Required Access Roles**, ensuring the database can filter results before they ever reach the LLM.

---

## 5. Query Routing with Semantic Router
Before a query hits the database, it passes through a **Semantic Router**. This layer performs two critical functions:
1.  **Domain Classification**: It identifies the intent of the query (e.g., is this a Finance question or an HR question?).
2.  **Access Firewall**: It compares the detected "Route" against the user's permitted roles. If a Marketing user asks a Finance question, the router blocks the request before a single vector search is performed.

---

## 6. Guardrails
FinSolve implements a "Defense in Depth" strategy using specialized guardrails.

### Input Guardrails
*   **Rate Limiting**: Prevents API abuse by tracking session-level query counts.
*   **PII Scrubbing**: Automatically detects and blocks queries containing sensitive numbers like Aadhar, PAN, or Bank account details.
*   **Prompt Injection Detection**: Blocks malicious attempts to bypass system instructions or extract hidden data.
*   **Topic Filtering**: Ensures the assistant stays focused on company-relevant queries.

### Output Guardrails
*   **Grounding Check**: Validates that the LLM's response is mathematically supported by the retrieved context.
*   **Role Leakage Protection**: A final security layer that audits the generated response to ensure no sensitive "out-of-role" information was hallucinated or leaked.
*   **Automated Citations**: Appends verified filenames and page numbers to the response for transparency.

---

## 7. Evaluations with RAGAS
To quantify system performance, we conduct **Ablation Studies** using the RAGAS framework.

*   **Metrics**: We track Faithfulness (grounding), Answer Relevancy, Context Precision, and Context Recall on a [0, 1] scale.
*   **Aggregated Scoring**: An "Overall Score" combines all metrics to provide a single health indicator for the pipeline.
*   **Infrastructure**: Evaluations are powered by `llama-3.1-8b-instant` on Groq, using asynchronous processing and rate-limit throttling to ensure reliable scoring.

---

## 8. The Application Interface

### a. Login Screen
A clean, secure entry point where employees log in using their corporate email. The system automatically fetches their role and access permissions from the HR database to customize the experience. For the purpose of this project, passwords have been excluded altogether for simplicity. In case of a real-world application, a robust authentication system must be implemented.

### b. Chat Interface
*   **Contextual Conversations**: A modern, responsive chat window with side-by-side history.
*   **User-Filtered History**: Chat history is private; users only see their own previous sessions.
*   **Guardrail Notifications**: Dynamic visual feedback when a guardrail is triggered (e.g., a "Privacy Notice" for PII).

### c. Admin Panel
A high-privilege dashboard for system administrators:
*   **Employee Management**: Full CRUD operations for managing the employee roster.
*   **Document Explorer**: View and manage the files within each department's collection.
*   **Database Reindexing**: A one-click trigger to clear the vector store and re-process all documents with updated logic or metadata.

---

*Built with Python (FastAPI/Docling/Qdrant) and React (Next.js/Lucide).*
