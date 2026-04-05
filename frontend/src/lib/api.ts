const API_BASE = "http://localhost:8000/api";

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function loginEmployee(email: string) {
  const res = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to login");
  }
  return res.json();
}

// ── Chat history ──────────────────────────────────────────────────────────────

export async function fetchHistory(employeeId: string) {
  const res = await fetch(`${API_BASE}/history?employee_id=${employeeId}`);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json() as Promise<{ session_id: string; first_query: string; last_updated: string }[]>;
}

export async function fetchSessionHistory(sessionId: string, employeeId: string) {
  const res = await fetch(`${API_BASE}/history/${sessionId}?employee_id=${employeeId}`);
  if (!res.ok) throw new Error("Failed to fetch session history");
  return res.json() as Promise<{ query: string; response: string; timestamp: string }[]>;
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export async function sendChatMessage(
  query: string,
  employeeId: string,
  sessionId?: string
) {
  const payload: Record<string, string> = { query, employee_id: employeeId };
  if (sessionId) payload.session_id = sessionId;

  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to send message");
  return res.json() as Promise<{
    session_id: string;
    response: string;
    user_roles: string[];
    guardrail_type: string | null;
    is_blocked: boolean;
  }>;
}

// ── Admin: employees ──────────────────────────────────────────────────────────

export async function adminListEmployees(
  adminId: string,
  search = "",
  page = 1,
  pageSize = 20
) {
  const params = new URLSearchParams({
    admin_id: adminId,
    search,
    page: String(page),
    page_size: String(pageSize),
  });
  const res = await fetch(`${API_BASE}/admin/employees?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to list employees");
  }
  return res.json() as Promise<{
    employees: Record<string, string>[];
    total: number;
    page: number;
    page_size: number;
  }>;
}

export async function adminCreateEmployee(adminId: string, data: {
  employee_id: string; name: string; email: string; role: string;
  department: string; access_role: string; gender?: string;
  designation_level?: string; employment_type?: string;
}) {
  const res = await fetch(`${API_BASE}/admin/employees?admin_id=${adminId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to create employee");
  }
  return res.json();
}

export async function adminUpdateEmployee(
  adminId: string,
  employeeId: string,
  data: { name?: string; email?: string; role?: string; department?: string; access_role?: string }
) {
  const res = await fetch(
    `${API_BASE}/admin/employees/${employeeId}?admin_id=${adminId}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to update employee");
  }
  return res.json();
}

export async function adminDeleteEmployee(adminId: string, employeeId: string) {
  const res = await fetch(
    `${API_BASE}/admin/employees/${employeeId}?admin_id=${adminId}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete employee");
  }
  return res.json();
}

// ── Admin: documents ──────────────────────────────────────────────────────────

export async function adminListDocuments(adminId: string) {
  const res = await fetch(`${API_BASE}/admin/documents?admin_id=${adminId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to list documents");
  }
  return res.json() as Promise<
    Record<string, { filename: string; collection: string; size_bytes: number; size_human: string }[]>
  >;
}

export async function adminDeleteDocument(
  adminId: string,
  collection: string,
  filename: string
) {
  const res = await fetch(
    `${API_BASE}/admin/documents/${collection}/${encodeURIComponent(filename)}?admin_id=${adminId}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete document");
  }
  return res.json();
}

export async function adminUploadDocument(
  adminId: string,
  collection: string,
  file: File
) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${API_BASE}/admin/documents/upload?admin_id=${adminId}&collection=${collection}`,
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload document");
  }
  return res.json();
}

// ── Admin: reindex ────────────────────────────────────────────────────────────

export async function adminTriggerReindex(adminId: string) {
  const res = await fetch(`${API_BASE}/admin/reindex?admin_id=${adminId}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start reindex");
  }
  return res.json();
}

export async function adminReindexStatus(adminId: string) {
  const res = await fetch(`${API_BASE}/admin/reindex/status?admin_id=${adminId}`);
  if (!res.ok) throw new Error("Failed to get reindex status");
  return res.json() as Promise<{
    running: boolean;
    message: string;
    progress: number;
    last_run: string | null;
  }>;
}
