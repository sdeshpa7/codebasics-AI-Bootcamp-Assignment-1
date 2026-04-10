"""
app/api.py
FastAPI Server for the FinSolve RAG Pipeline.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import uuid
import csv
import os
import threading
from datetime import datetime

from app.database import init_db, get_connection
from app.config import settings
from run import answer

app = FastAPI(title="FinSolve API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()


# ── Employee CSV helpers ─────────────────────────────────────────────────────

EMPLOYEES_CSV_PATH = "data/hr/employees.csv"

# Admins are identified by email — NOT by access_role.
# Their access_role (e.g. Engineering) still governs which documents they can query.
# Admin powers: create roles, manage logins, update documents.
ADMIN_EMAILS: Set[str] = {
    "karthik.kuppuswamy@finsolve.com",
    "rajesh.tiwari@finsolve.com",
    "chitra.aggarwal@finsolve.com",
    "krishnan.saxena@finsolve.com",
    "karthik.chopra@finsolve.com",
}


def _read_employees() -> list[dict]:
    if not os.path.exists(EMPLOYEES_CSV_PATH):
        return []
    with open(EMPLOYEES_CSV_PATH, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_employees(employees: list[dict]) -> None:
    if not employees:
        return
    fieldnames = list(employees[0].keys())
    with open(EMPLOYEES_CSV_PATH, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(employees)


def lookup_employee(key: str, value: str) -> Optional[dict]:
    for emp in _read_employees():
        if emp.get(key) == value:
            return emp
    return None


def _require_admin(admin_id: str) -> dict:
    emp = lookup_employee("employee_id", admin_id)
    if not emp:
        raise HTTPException(status_code=401, detail="Invalid employee ID")
    if emp.get("email") not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return emp


# ── Guardrail detection ──────────────────────────────────────────────────────

def _detect_guardrail(response: str) -> tuple[Optional[str], bool]:
    """
    Parse the response text to identify guardrail events.
    Returns (guardrail_type, is_blocked).
    """
    if response.startswith("🚫"):
        return "rbac_blocked", True
    if response.startswith("😊"):
        return "off_topic", True
    if "reached the session limit" in response:
        return "rate_limit", True
    if response.startswith("🔒 Your query appears to contain"):
        return "pii_blocked", True
    if response.startswith("🔒 Your query references"):
        return "pii_hr_sensitive", True
    if "I couldn't find any relevant documents" in response:
        return "no_results", False
    has_output_warning = (
        "⚠️ **Grounding Notice**" in response
        or "🔒 **Access Notice**" in response
        or "📌 **Citation Notice**" in response
    )
    if has_output_warning:
        return "output_warning", False
    return None, False


# ── Schemas ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str

class LoginResponse(BaseModel):
    employee_id: str
    name: str
    email: str
    access_role: str
    department: str
    is_admin: bool = False

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    employee_id: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    user_roles: List[str]
    guardrail_type: Optional[str] = None
    is_blocked: bool = False

class EmployeeCreateRequest(BaseModel):
    employee_id: str
    name: str
    email: str
    role: str
    department: str
    access_role: str
    gender: str = "Unknown"
    designation_level: str = "Mid"
    employment_type: str = "Full-Time"

class EmployeeUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    access_role: Optional[str] = None


# ── Reindex background job ────────────────────────────────────────────────────

_reindex_status: Dict[str, Any] = {
    "running": False,
    "message": "Idle — no reindex has been run yet.",
    "progress": 0,
    "last_run": None,
}
_reindex_lock = threading.Lock()

ALLOWED_ROLES       = ["C-Level", "HR", "Finance", "Engineering", "Marketing", "General"]
ALLOWED_COLLECTIONS = ["general", "hr", "finance", "engineering", "marketing"]


def _run_reindex() -> None:
    global _reindex_status
    with _reindex_lock:
        try:
            from app.ingestion import ingest_all
            from app.chunking import chunk_all
            from app.embeddings import embed_chunks
            from app.vector_store import upsert_embeddings, get_client

            _reindex_status.update({"message": "Clearing existing Qdrant collection…", "progress": 5})
            client = get_client()
            try:
                client.delete_collection(settings.qdrant_collection)
            except Exception:
                pass  # collection may not exist yet

            _reindex_status.update({"message": "Parsing & ingesting documents…", "progress": 20})
            result = ingest_all()

            _reindex_status.update({"message": "Chunking documents…", "progress": 45})
            chunks = chunk_all(result["documents"])

            _reindex_status.update({"message": f"Embedding {len(chunks)} chunks…", "progress": 65})
            embedded = embed_chunks(chunks)

            _reindex_status.update({"message": "Upserting vectors to Qdrant…", "progress": 85})
            upsert_embeddings(embedded)

            _reindex_status = {
                "running": False,
                "message": f"Done! {len(embedded)} chunks indexed from {len(result['documents'])} documents.",
                "progress": 100,
                "last_run": datetime.now().isoformat(),
            }
        except Exception as e:
            _reindex_status = {
                "running": False,
                "message": f"Error during reindex: {e}",
                "progress": 0,
                "last_run": None,
            }


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.post("/api/login", response_model=LoginResponse)
def login_endpoint(req: LoginRequest):
    employee = lookup_employee("email", req.email)
    if not employee:
        raise HTTPException(status_code=401, detail="incorrect email id, please try again")
    return LoginResponse(
        employee_id=employee["employee_id"],
        name=employee["name"],
        email=employee["email"],
        access_role=employee["access_role"],
        department=employee["department"],
        is_admin=employee["email"] in ADMIN_EMAILS,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    employee = lookup_employee("employee_id", req.employee_id)
    if not employee:
        raise HTTPException(status_code=401, detail="Invalid employee ID")

    emp_role   = employee["access_role"]
    user_roles = list(set(settings.user_common_roles + [emp_role]))
    session_id = req.session_id or str(uuid.uuid4())

    try:
        res = answer(
            query=req.query, 
            session_id=session_id, 
            user_access_roles=user_roles,
            employment_type=employee.get("employment_type", "Full-Time"),
            employee_id=employee["employee_id"]
        )
        guardrail_type, is_blocked = _detect_guardrail(res)
        return ChatResponse(
            session_id=session_id,
            response=res,
            user_roles=user_roles,
            guardrail_type=guardrail_type,
            is_blocked=is_blocked,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_history_sessions(employee_id: str):
    """Return unique sessions with first_query title and last_updated timestamp."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT session_id,
                   MIN(user_query) AS first_query,
                   MAX(timestamp)  AS last_updated
            FROM chat_history
            WHERE employee_id = ?
            GROUP BY session_id
            ORDER BY last_updated DESC
            LIMIT 50
            """,
            (employee_id,)
        )
        rows = cur.fetchall()
        return [
            {
                "session_id":   r["session_id"],
                "first_query":  r["first_query"],
                "last_updated": r["last_updated"],
            }
            for r in rows
        ]


@app.get("/api/history/{session_id}")
def get_session_history(session_id: str, employee_id: str):
    """Return the full message thread for a session."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT user_query, llm_response, timestamp
            FROM chat_history
            WHERE session_id = ? AND employee_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id, employee_id),
        )
        rows = cur.fetchall()
        return [
            {"query": r["user_query"], "response": r["llm_response"], "timestamp": r["timestamp"]}
            for r in rows
        ]


# ── Admin: employees ──────────────────────────────────────────────────────────

@app.get("/api/admin/employees")
def admin_list_employees(admin_id: str, search: str = "", page: int = 1, page_size: int = 20):
    _require_admin(admin_id)
    employees = _read_employees()
    if search:
        sl = search.lower()
        employees = [
            e for e in employees
            if sl in e.get("name", "").lower()
            or sl in e.get("email", "").lower()
            or sl in e.get("department", "").lower()
            or sl in e.get("access_role", "").lower()
            or sl in e.get("employee_id", "").lower()
        ]
    total    = len(employees)
    start    = (page - 1) * page_size
    paginated = employees[start:start + page_size]
    return {"employees": paginated, "total": total, "page": page, "page_size": page_size}


@app.post("/api/admin/employees")
def admin_create_employee(admin_id: str, req: EmployeeCreateRequest):
    _require_admin(admin_id)
    employees = _read_employees()

    if any(e["employee_id"] == req.employee_id for e in employees):
        raise HTTPException(status_code=400, detail="Employee ID already exists")
    if any(e["email"] == req.email for e in employees):
        raise HTTPException(status_code=400, detail="Email already in use")
    if req.access_role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid access_role. Allowed: {ALLOWED_ROLES}")

    fieldnames = list(employees[0].keys()) if employees else [
        "employee_id", "name", "gender", "role", "department", "access_role",
        "designation_level", "email", "phone", "location", "date_of_birth",
        "date_of_joining", "employment_type", "employment_status", "manager_id",
        "salary", "leave_balance", "leaves_taken", "attendance_pct",
        "performance_rating", "last_review_date", "exit_date",
    ]
    new_emp = {k: "" for k in fieldnames}
    new_emp.update({
        "employee_id":       req.employee_id,
        "name":              req.name,
        "email":             req.email,
        "role":              req.role,
        "department":        req.department,
        "access_role":       req.access_role,
        "gender":            req.gender,
        "designation_level": req.designation_level,
        "employment_type":   req.employment_type,
        "employment_status": "Active",
    })
    employees.append(new_emp)
    _write_employees(employees)
    return {"message": f"Employee {req.employee_id} created successfully"}


@app.put("/api/admin/employees/{employee_id}")
def admin_update_employee(admin_id: str, employee_id: str, req: EmployeeUpdateRequest):
    _require_admin(admin_id)
    employees = _read_employees()
    found = False
    for emp in employees:
        if emp["employee_id"] == employee_id:
            if req.name        is not None: emp["name"]        = req.name
            if req.email       is not None: emp["email"]       = req.email
            if req.role        is not None: emp["role"]        = req.role
            if req.department  is not None: emp["department"]  = req.department
            if req.access_role is not None:
                if req.access_role not in ALLOWED_ROLES:
                    raise HTTPException(status_code=400, detail=f"Invalid access_role. Allowed: {ALLOWED_ROLES}")
                emp["access_role"] = req.access_role
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Employee not found")
    _write_employees(employees)
    return {"message": f"Employee {employee_id} updated"}


@app.delete("/api/admin/employees/{employee_id}")
def admin_delete_employee(admin_id: str, employee_id: str):
    _require_admin(admin_id)
    if employee_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    employees = _read_employees()
    filtered = [e for e in employees if e["employee_id"] != employee_id]
    if len(filtered) == len(employees):
        raise HTTPException(status_code=404, detail="Employee not found")
    _write_employees(filtered)
    return {"message": f"Employee {employee_id} deleted"}


# ── Admin: documents ──────────────────────────────────────────────────────────

@app.get("/api/admin/documents")
def admin_list_documents(admin_id: str):
    _require_admin(admin_id)
    result: Dict[str, list] = {}
    for collection in ALLOWED_COLLECTIONS:
        folder = os.path.join(settings.data_dir, collection)
        files: list = []
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and not fname.startswith("."):
                    size = os.path.getsize(fpath)
                    files.append({
                        "filename":   fname,
                        "collection": collection,
                        "size_bytes": size,
                        "size_human": (
                            f"{size // (1024*1024)} MB" if size >= 1024 * 1024
                            else f"{size // 1024} KB"
                        ),
                    })
        result[collection] = files
    return result


@app.delete("/api/admin/documents/{collection}/{filename}")
def admin_delete_document(admin_id: str, collection: str, filename: str):
    _require_admin(admin_id)
    if collection not in ALLOWED_COLLECTIONS:
        raise HTTPException(status_code=400, detail="Invalid collection")
    filepath = os.path.join(settings.data_dir, collection, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(filepath)
    return {"message": f"Deleted {filename} from {collection}"}


@app.post("/api/admin/documents/upload")
async def admin_upload_document(admin_id: str, collection: str, file: UploadFile = File(...)):
    _require_admin(admin_id)
    if collection not in ALLOWED_COLLECTIONS:
        raise HTTPException(status_code=400, detail="Invalid collection")
    folder = os.path.join(settings.data_dir, collection)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    size_kb = len(content) // 1024
    return {"message": f"Uploaded {file.filename} to {collection} ({size_kb} KB)"}


# ── Admin: reindex ────────────────────────────────────────────────────────────

@app.post("/api/admin/reindex")
def admin_trigger_reindex(background_tasks: BackgroundTasks, admin_id: str):
    _require_admin(admin_id)
    global _reindex_status
    if _reindex_status.get("running"):
        raise HTTPException(status_code=409, detail="Reindex already in progress")
    _reindex_status = {
        "running":  True,
        "message":  "Starting reindex…",
        "progress": 0,
        "last_run": None,
    }
    background_tasks.add_task(_run_reindex)
    return {"message": "Reindex started in background"}


@app.get("/api/admin/reindex/status")
def admin_reindex_status(admin_id: str):
    _require_admin(admin_id)
    return _reindex_status
