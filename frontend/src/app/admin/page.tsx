"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Users, ArrowLeft, Plus, Trash2,
  Search, Edit2, Check, X, ChevronLeft,
  ChevronRight, Shield, AlertCircle, CheckCircle,
  Loader2, KeyRound, MessageSquare, LogOut, Upload, ChevronDown,
} from "lucide-react";
import {
  adminListEmployees, adminCreateEmployee, adminUpdateEmployee,
  adminDeleteEmployee, adminUploadDocument,
} from "@/lib/api";
import { loginEmployee } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type Employee = Record<string, string>;

// ── Constants ─────────────────────────────────────────────────────────────────

const ROLES = ["C-Level", "HR", "Finance", "Engineering", "Marketing", "General"];
const COLLECTIONS = [
  { label: "Engineering", value: "engineering" },
  { label: "Finance",     value: "finance" },
  { label: "General",     value: "general" },
  { label: "HR",          value: "hr" },
  { label: "Marketing",   value: "marketing" },
];

// ── Helper ────────────────────────────────────────────────────────────────────

function Pill({ label, cls }: { label: string; cls: string }) {
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-semibold ${cls}`}>
      {label}
    </span>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AdminPage() {
  const router = useRouter();

  // Auth
  const [authStatus, setAuthStatus] = useState<"loading" | "authorised" | "unauthorised">("loading");
  const [adminId, setAdminId] = useState<string | null>(null);
  const [adminName, setAdminName] = useState<string>("");
  const [loginEmail, setLoginEmail] = useState("");
  const [loginError, setLoginError] = useState("");
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // Toast
  const [toast, setToast] = useState<{ msg: string; type: "ok" | "err" } | null>(null);

  // ── Users state ─────────────────────────────────────────────────────────────
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [empTotal, setEmpTotal] = useState(0);
  const [empPage, setEmpPage] = useState(1);
  const [empSearch, setEmpSearch] = useState("");
  const [empLoading, setEmpLoading] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editData, setEditData] = useState<Partial<Employee>>({});
  const [showAddForm, setShowAddForm] = useState(false);
  const [newEmp, setNewEmp] = useState({
    employee_id: "", name: "", email: "", role: "", department: "",
    access_role: "General", gender: "Unknown", designation_level: "Mid",
  });
  const PAGE_SIZE = 15;

  // ── Document upload state ────────────────────────────────────────────────────
  const [showDocDropdown, setShowDocDropdown] = useState(false);
  const [uploadingCollection, setUploadingCollection] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (docDropdownRef.current && !docDropdownRef.current.contains(e.target as Node)) {
        setShowDocDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // ── Load session from sessionStorage ────────────────────────────────────────
  useEffect(() => {
    const raw = sessionStorage.getItem("finsolve_admin");
    if (!raw) { setAuthStatus("unauthorised"); return; }
    try {
      const u = JSON.parse(raw);
      if (!u.is_admin) { setAuthStatus("unauthorised"); return; }
      setAdminId(u.employee_id);
      setAdminName(u.name);
      setAuthStatus("authorised");
    } catch {
      setAuthStatus("unauthorised");
    }
  }, []);

  // Auto-load employees once authorised
  useEffect(() => {
    if (authStatus === "authorised" && adminId) loadEmployees();
  }, [authStatus, adminId]);

  // Auto-dismiss toast
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(t);
  }, [toast]);

  const showToast = (msg: string, type: "ok" | "err" = "ok") => setToast({ msg, type });

  // ── Admin login ──────────────────────────────────────────────────────────────

  const handleAdminLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginEmail.trim()) return;
    setIsLoggingIn(true);
    setLoginError("");
    try {
      const data = await loginEmployee(loginEmail.trim());
      if (!data.is_admin) {
        // Redirect to main login with a clear error — do not grant any session
        router.push("/?error=not_admin");
        return;
      }
      // Store in a separate key so it doesn't collide with chat session
      sessionStorage.setItem("finsolve_admin", JSON.stringify(data));
      // Also store as main user so chat interface picks up the session
      sessionStorage.setItem("finsolve_user", JSON.stringify(data));
      setAdminId(data.employee_id);
      setAdminName(data.name);
      setAuthStatus("authorised");
    } catch (e: any) {
      setLoginError(e.message || "Incorrect email, please try again.");
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = () => {
    // Clear both admin and main chat sessions so the user is fully signed out
    sessionStorage.removeItem("finsolve_admin");
    sessionStorage.removeItem("finsolve_user");
    setAuthStatus("unauthorised");
    setAdminId(null);
    setAdminName("");
    setLoginEmail("");
    setLoginError("");
  };

  // ── Employees ────────────────────────────────────────────────────────────────

  const loadEmployees = async (page = empPage, search = empSearch) => {
    if (!adminId) return;
    setEmpLoading(true);
    try {
      const res = await adminListEmployees(adminId, search, page, PAGE_SIZE);
      setEmployees(res.employees);
      setEmpTotal(res.total);
      setEmpPage(page);
    } catch (e: any) {
      showToast(e.message, "err");
    } finally {
      setEmpLoading(false);
    }
  };

  const handleSearch = (v: string) => {
    setEmpSearch(v);
    loadEmployees(1, v);
  };

  const startEdit = (emp: Employee) => {
    setEditingId(emp.employee_id);
    setEditData({ name: emp.name, email: emp.email, role: emp.role, department: emp.department, access_role: emp.access_role });
  };

  const cancelEdit = () => { setEditingId(null); setEditData({}); };

  const saveEdit = async (id: string) => {
    if (!adminId) return;
    try {
      await adminUpdateEmployee(adminId, id, editData as any);
      showToast("Employee updated");
      cancelEdit();
      loadEmployees();
    } catch (e: any) {
      showToast(e.message, "err");
    }
  };

  const deleteEmp = async (id: string) => {
    if (!adminId || !confirm(`Delete employee ${id}?`)) return;
    try {
      await adminDeleteEmployee(adminId, id);
      showToast("Employee deleted");
      loadEmployees();
    } catch (e: any) {
      showToast(e.message, "err");
    }
  };

  const createEmp = async () => {
    if (!adminId) return;
    try {
      await adminCreateEmployee(adminId, newEmp);
      showToast(`Employee ${newEmp.employee_id} created`);
      setShowAddForm(false);
      setNewEmp({ employee_id: "", name: "", email: "", role: "", department: "", access_role: "General", gender: "Unknown", designation_level: "Mid" });
      loadEmployees();
    } catch (e: any) {
      showToast(e.message, "err");
    }
  };

  const totalPages = Math.ceil(empTotal / PAGE_SIZE);

  // ── Loading splash ────────────────────────────────────────────────────────────

  if (authStatus === "loading") return null;

  // ── Admin Login Screen ────────────────────────────────────────────────────────

  if (authStatus === "unauthorised") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-50 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 to-blue-50 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-indigo-500/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="w-full max-w-md relative z-10 px-4">
          <div className="bg-white rounded-3xl shadow-2xl border border-zinc-100 p-8">

            {/* Header */}
            <div className="flex flex-col items-center mb-8">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-blue-600 flex items-center justify-center text-white shadow-xl shadow-indigo-500/30 mb-4">
                <Shield className="w-7 h-7" />
              </div>
              <h1 className="text-2xl font-bold text-zinc-900">Admin Portal</h1>
              <p className="text-sm text-zinc-500 mt-2 text-center leading-relaxed">
                Restricted to authorised FinSolve administrators.<br />
                Enter your admin email address to sign in.
              </p>
            </div>

            {/* Login form */}
            <form onSubmit={handleAdminLogin} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-sm font-semibold text-zinc-700">Admin Email</label>
                <Input
                  type="email"
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                  placeholder="e.g. karthik.kuppuswamy@finsolve.com"
                  className="h-12 bg-zinc-50 border-zinc-200 focus-visible:ring-indigo-500 px-4"
                  autoComplete="email"
                  required
                />
              </div>

              {loginError && (
                <div className="text-sm text-red-600 font-medium flex items-center gap-1.5 bg-red-50 px-3 py-2 rounded-lg border border-red-100">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  {loginError}
                </div>
              )}

              <Button
                type="submit"
                className="w-full h-12 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl gap-2 shadow-md shadow-indigo-600/20 transition-all active:scale-[0.98]"
                disabled={isLoggingIn}
              >
                {isLoggingIn ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Verifying…</>
                ) : (
                  <><KeyRound className="w-4 h-4" /> Sign In to Admin Portal</>
                )}
              </Button>
            </form>

            {/* Back to Chat Interface — same style as Go to Admin Portal */}
            <div className="relative flex items-center gap-3 pt-1">
              <div className="flex-1 h-px bg-zinc-100" />
              <span className="text-[11px] text-zinc-400 font-medium">or</span>
              <div className="flex-1 h-px bg-zinc-100" />
            </div>
            <button
              type="button"
              onClick={() => router.push("/")}
              className="w-full h-11 flex items-center justify-center gap-2 rounded-xl border border-indigo-200 bg-indigo-50 text-indigo-700 text-sm font-medium hover:bg-indigo-100 transition-all active:scale-[0.98]"
            >
              <MessageSquare className="w-4 h-4" />
              Go to Chat Interface
            </button>

            <p className="text-center text-[10px] text-zinc-400 mt-5 leading-relaxed">
              Your document access remains governed by your assigned role.<br />
              Admin privileges cover user management only.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Admin Dashboard ───────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#f8f9fb] font-sans">

      {/* Toast */}
      {toast && (
        <div className={`fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 rounded-xl shadow-lg border text-sm font-medium animate-in fade-in slide-in-from-top-2 ${
          toast.type === "ok"
            ? "bg-white border-green-200 text-green-800"
            : "bg-white border-red-200 text-red-700"
        }`}>
          {toast.type === "ok"
            ? <CheckCircle className="w-4 h-4 text-green-500" />
            : <AlertCircle className="w-4 h-4 text-red-500" />}
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <header className="h-14 bg-white border-b shadow-sm flex items-center px-6 gap-4 sticky top-0 z-40">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-indigo-600" />
          <span className="font-semibold text-zinc-800">Admin Panel</span>
          <span className="text-[10px] px-1.5 py-0.5 bg-yellow-100 text-yellow-800 border border-yellow-200 rounded font-bold uppercase tracking-wide">
            Admin
          </span>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-zinc-400">
            Signed in as <span className="font-semibold text-zinc-600">{adminName}</span>
          </span>

          {/* Go to Chat Interface */}
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 px-3 py-1.5 rounded-lg transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5" />
            Chat Interface
          </button>

          {/* Logout */}
          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 hover:text-red-600 hover:bg-red-50 px-3 py-1.5 rounded-lg transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign Out
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* Page title */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
              <Users className="w-5 h-5 text-indigo-600" />
              User Management
            </h2>
            <p className="text-sm text-zinc-500 mt-0.5">{empTotal} employees in the system</p>
          </div>
          <div className="flex items-center gap-2">
            {/* Add Employee */}
            <Button
              onClick={() => setShowAddForm(!showAddForm)}
              className="bg-indigo-600 hover:bg-indigo-700 text-white gap-2 h-9"
            >
              <Plus className="w-4 h-4" /> Add Employee
            </Button>

            {/* Add Document */}
            <div className="relative" ref={docDropdownRef}>
              <Button
                onClick={() => setShowDocDropdown((v) => !v)}
                variant="outline"
                className="gap-2 h-9 border-indigo-200 text-indigo-700 hover:bg-indigo-50"
              >
                <Upload className="w-4 h-4" /> Add Document
                <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showDocDropdown ? "rotate-180" : ""}`} />
              </Button>

              {showDocDropdown && (
                <div className="absolute right-0 mt-1.5 w-44 bg-white border border-zinc-200 rounded-xl shadow-lg z-30 py-1 animate-in fade-in zoom-in-95 duration-100">
                  <p className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wide px-3 pt-2 pb-1">Select Collection</p>
                  {COLLECTIONS.map((col) => (
                    <button
                      key={col.value}
                      onClick={() => {
                        setUploadingCollection(col.value);
                        setShowDocDropdown(false);
                        // Trigger file picker
                        if (fileInputRef.current) {
                          fileInputRef.current.value = "";
                          fileInputRef.current.click();
                        }
                      }}
                      className="w-full text-left px-3 py-2 text-sm text-zinc-700 hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
                    >
                      {col.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Hidden file input — triggered programmatically */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.md,.txt,.csv"
                className="hidden"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file || !adminId || !uploadingCollection) return;
                  try {
                    await adminUploadDocument(adminId, uploadingCollection, file);
                    showToast(`"${file.name}" uploaded to ${uploadingCollection}`);
                  } catch (err: any) {
                    showToast(err.message || "Upload failed", "err");
                  } finally {
                    setUploadingCollection(null);
                  }
                }}
              />
            </div>
          </div>
        </div>

        {/* ── Add form ──────────────────────────────────────────────────────────── */}
        {showAddForm && (
          <div className="bg-white rounded-xl border shadow-sm p-5 mb-4 animate-in fade-in slide-in-from-top-2">
            <h3 className="font-semibold text-zinc-800 mb-4 flex items-center gap-2">
              <Plus className="w-4 h-4 text-indigo-600" /> New Employee
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { key: "employee_id", label: "Employee ID",  placeholder: "FINEMP9999" },
                { key: "name",        label: "Full Name",    placeholder: "Jane Smith" },
                { key: "email",       label: "Email",        placeholder: "jane@finsolve.com" },
                { key: "role",        label: "Job Title",    placeholder: "Data Analyst" },
                { key: "department",  label: "Department",   placeholder: "Finance" },
              ].map((f) => (
                <div key={f.key}>
                  <label className="text-xs font-semibold text-zinc-600 block mb-1">{f.label}</label>
                  <Input
                    value={(newEmp as any)[f.key]}
                    onChange={(e) => setNewEmp((p) => ({ ...p, [f.key]: e.target.value }))}
                    placeholder={f.placeholder}
                    className="h-8 text-sm"
                  />
                </div>
              ))}
              <div>
                <label className="text-xs font-semibold text-zinc-600 block mb-1">Access Role</label>
                <select
                  value={newEmp.access_role}
                  onChange={(e) => setNewEmp((p) => ({ ...p, access_role: e.target.value }))}
                  className="h-8 w-full text-sm border border-zinc-200 rounded-md px-2 bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {ROLES.map((r) => <option key={r}>{r}</option>)}
                </select>
              </div>
            </div>
            <div className="flex gap-2 mt-4">
              <Button onClick={createEmp} className="bg-indigo-600 hover:bg-indigo-700 text-white h-8 text-sm gap-1">
                <Check className="w-3.5 h-3.5" /> Create
              </Button>
              <Button onClick={() => setShowAddForm(false)} variant="outline" className="h-8 text-sm gap-1">
                <X className="w-3.5 h-3.5" /> Cancel
              </Button>
            </div>
          </div>
        )}

        {/* ── Search ────────────────────────────────────────────────────────────── */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-400" />
          <Input
            value={empSearch}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search by name, email, department, role…"
            className="pl-9 h-9 text-sm bg-white"
          />
        </div>

        {/* ── Employee table ─────────────────────────────────────────────────────── */}
        <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-zinc-50">
                {["ID", "Name", "Email", "Role / Dept", "Access Role", "Actions"].map((h) => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-semibold text-zinc-500 uppercase tracking-wide whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {empLoading ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-zinc-400">
                    <Loader2 className="w-5 h-5 animate-spin inline mr-2" />Loading…
                  </td>
                </tr>
              ) : employees.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-zinc-400 text-sm">
                    No employees found.
                  </td>
                </tr>
              ) : employees.map((emp) => (
                <tr key={emp.employee_id} className="border-b hover:bg-zinc-50 transition-colors">
                  <td className="px-4 py-3 font-mono text-xs text-zinc-500">{emp.employee_id}</td>

                  {/* Name */}
                  <td className="px-4 py-3">
                    {editingId === emp.employee_id ? (
                      <Input value={editData.name ?? ""} onChange={(e) => setEditData((p) => ({ ...p, name: e.target.value }))} className="h-7 text-xs" />
                    ) : (
                      <span className="font-medium text-zinc-800">{emp.name}</span>
                    )}
                  </td>

                  {/* Email */}
                  <td className="px-4 py-3 text-zinc-500 text-xs">
                    {editingId === emp.employee_id ? (
                      <Input value={editData.email ?? ""} onChange={(e) => setEditData((p) => ({ ...p, email: e.target.value }))} className="h-7 text-xs" />
                    ) : emp.email}
                  </td>

                  {/* Role / Dept */}
                  <td className="px-4 py-3 text-xs text-zinc-600">
                    {editingId === emp.employee_id ? (
                      <div className="flex gap-1">
                        <Input value={editData.role ?? ""} onChange={(e) => setEditData((p) => ({ ...p, role: e.target.value }))} placeholder="Role" className="h-7 text-xs w-24" />
                        <Input value={editData.department ?? ""} onChange={(e) => setEditData((p) => ({ ...p, department: e.target.value }))} placeholder="Dept" className="h-7 text-xs w-24" />
                      </div>
                    ) : (
                      <span>{emp.role}<span className="text-zinc-400"> · {emp.department}</span></span>
                    )}
                  </td>

                  {/* Access Role */}
                  <td className="px-4 py-3">
                    {editingId === emp.employee_id ? (
                      <select
                        value={editData.access_role ?? emp.access_role}
                        onChange={(e) => setEditData((p) => ({ ...p, access_role: e.target.value }))}
                        className="h-7 text-xs border border-zinc-200 rounded px-1 bg-white"
                      >
                        {ROLES.map((r) => <option key={r}>{r}</option>)}
                      </select>
                    ) : (
                      <Pill label={emp.access_role} cls="bg-indigo-50 text-indigo-700 border-indigo-200" />
                    )}
                  </td>

                  {/* Actions */}
                  <td className="px-4 py-3">
                    {editingId === emp.employee_id ? (
                      <div className="flex gap-1">
                        <button onClick={() => saveEdit(emp.employee_id)} className="p-1 text-green-600 hover:bg-green-50 rounded"><Check className="w-3.5 h-3.5" /></button>
                        <button onClick={cancelEdit} className="p-1 text-zinc-400 hover:bg-zinc-100 rounded"><X className="w-3.5 h-3.5" /></button>
                      </div>
                    ) : (
                      <div className="flex gap-1">
                        <button
                          onClick={() => startEdit(emp)}
                          title="Edit employee"
                          className="p-1 text-zinc-400 hover:text-indigo-600 hover:bg-indigo-50 rounded transition-colors"
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => deleteEmp(emp.employee_id)}
                          title="Delete employee"
                          className="p-1 text-zinc-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t bg-zinc-50 text-xs text-zinc-500">
              <span>Page {empPage} of {totalPages} ({empTotal} total)</span>
              <div className="flex gap-1">
                <button disabled={empPage <= 1} onClick={() => loadEmployees(empPage - 1)} className="p-1 rounded hover:bg-zinc-200 disabled:opacity-30">
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button disabled={empPage >= totalPages} onClick={() => loadEmployees(empPage + 1)} className="p-1 rounded hover:bg-zinc-200 disabled:opacity-30">
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Notice */}
        <p className="text-xs text-zinc-400 mt-4 text-center leading-relaxed">
          Assigning an access role controls which knowledge collections a user can query.<br />
          Admin privileges are managed separately and do not affect document access.
        </p>
      </div>
    </div>
  );
}
