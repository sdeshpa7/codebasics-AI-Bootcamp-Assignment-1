"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Send, PlusCircle, MessageSquare, Info, KeyRound,
  AlertCircle, LogOut, Shield, ChevronRight, Database,
  Lock, AlertTriangle, CheckCircle, XCircle, Search,
} from "lucide-react";
import {
  fetchHistory, fetchSessionHistory, sendChatMessage, loginEmployee,
} from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  guardrail_type?: string | null;
  is_blocked?: boolean;
};

type Session = {
  session_id: string;
  first_query: string;
  last_updated: string;
};

type User = {
  employee_id: string;
  name: string;
  email: string;
  access_role: string;
  department: string;
  is_admin: boolean;
};

// ── Constants ─────────────────────────────────────────────────────────────────

const ROLE_COLLECTIONS: Record<string, string[]> = {
  "C-Level":     ["general", "finance", "hr", "engineering", "marketing"],
  "HR":          ["general", "hr"],
  "Finance":     ["general", "finance"],
  "Engineering": ["general", "engineering"],
  "Marketing":   ["general", "marketing"],
  "General":     ["general"],
};

const COLLECTION_CONFIG: Record<string, { label: string; cls: string }> = {
  general:     { label: "General",     cls: "bg-slate-100 text-slate-700 border-slate-300"  },
  finance:     { label: "Finance",     cls: "bg-emerald-100 text-emerald-700 border-emerald-300" },
  hr:          { label: "HR",          cls: "bg-purple-100 text-purple-700 border-purple-300"  },
  engineering: { label: "Engineering", cls: "bg-blue-100 text-blue-700 border-blue-300"       },
  marketing:   { label: "Marketing",   cls: "bg-orange-100 text-orange-700 border-orange-300"  },
};

const ROLE_BADGE: Record<string, string> = {
  "C-Level":     "bg-yellow-100 text-yellow-800 border-yellow-300",
  "HR":          "bg-purple-100 text-purple-800 border-purple-300",
  "Finance":     "bg-emerald-100 text-emerald-800 border-emerald-300",
  "Engineering": "bg-blue-100 text-blue-800 border-blue-300",
  "Marketing":   "bg-orange-100 text-orange-800 border-orange-300",
  "General":     "bg-slate-100 text-slate-800 border-slate-300",
};

const GUARDRAIL_CONFIG: Record<string, {
  title: string; desc: string; cls: string; icon: React.ReactNode;
}> = {
  rbac_blocked: {
    title: "Access Denied",
    desc: "Your role does not permit access to this topic.",
    cls: "border-red-200 bg-red-50 text-red-800",
    icon: <XCircle className="w-4 h-4 text-red-600 shrink-0" />,
  },
  off_topic: {
    title: "Off-Topic Query",
    desc: "This question is outside FinSolve's knowledge domains.",
    cls: "border-amber-200 bg-amber-50 text-amber-800",
    icon: <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />,
  },
  rate_limit: {
    title: "Rate Limit Reached",
    desc: "You've reached the maximum queries for this session.",
    cls: "border-amber-200 bg-amber-50 text-amber-800",
    icon: <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />,
  },
  pii_blocked: {
    title: "Sensitive Data Detected",
    desc: "Your query contains personal identification numbers (Aadhar, PAN, bank account).",
    cls: "border-red-200 bg-red-50 text-red-800",
    icon: <Lock className="w-4 h-4 text-red-600 shrink-0" />,
  },
  pii_hr_sensitive: {
    title: "HR-Restricted Data",
    desc: "Only HR personnel can query personal employee records.",
    cls: "border-orange-200 bg-orange-50 text-orange-800",
    icon: <Lock className="w-4 h-4 text-orange-600 shrink-0" />,
  },
  no_results: {
    title: "No Matching Documents",
    desc: "The knowledge base has no relevant content for this query.",
    cls: "border-zinc-200 bg-zinc-50 text-zinc-700",
    icon: <Search className="w-4 h-4 text-zinc-500 shrink-0" />,
  },
  output_warning: {
    title: "Response Quality Notice",
    desc: "The response includes grounding, citation, or access warnings — review carefully.",
    cls: "border-yellow-200 bg-yellow-50 text-yellow-800",
    icon: <AlertCircle className="w-4 h-4 text-yellow-600 shrink-0" />,
  },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function getInitials(name: string) {
  return name.split(" ").map((n) => n[0]).join("").toUpperCase();
}

function truncate(str: string, len: number) {
  if (!str) return "Untitled session";
  return str.length > len ? str.slice(0, len) + "…" : str;
}

function getAccessibleCollections(role: string): string[] {
  return ROLE_COLLECTIONS[role] ?? ["general"];
}

// ── Sub-components ────────────────────────────────────────────────────────────

function GuardrailBanner({ type, isBlocked }: { type: string; isBlocked: boolean }) {
  const cfg = GUARDRAIL_CONFIG[type];
  if (!cfg) return null;
  return (
    <div className={`flex items-start gap-2 px-3 py-2 rounded-lg border text-xs font-medium mb-2 ${cfg.cls}`}>
      {cfg.icon}
      <div>
        <span className="font-semibold">{cfg.title}</span>
        {" · "}
        <span className="opacity-80">{cfg.desc}</span>
        {isBlocked && (
          <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-red-200/60 text-red-800 font-bold uppercase tracking-wide">
            Blocked
          </span>
        )}
      </div>
    </div>
  );
}

function CollectionBadge({ collection }: { collection: string }) {
  const cfg = COLLECTION_CONFIG[collection];
  if (!cfg) return null;
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

import { Suspense } from "react";

// ── Main component ────────────────────────────────────────────────────────────

function FinSolveDashboard() {
  const router = useRouter();

  // Auth
  const searchParams = useSearchParams();
  const [user, setUser] = useState<User | null>(null);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginError, setLoginError] = useState("");
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // Chat
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentSessionTitle, setCurrentSessionTitle] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputVal, setInputVal] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  // Restore session from sessionStorage on mount (e.g. coming from Admin Portal)
  // Also read any error param passed via redirect from Admin Portal
  useEffect(() => {
    const raw = sessionStorage.getItem("finsolve_user");
    if (!raw) {
      // Check if redirected from Admin Portal with an error
      if (searchParams.get("error") === "not_admin") {
        setLoginError("You do not have admin status. Please use the regular login below.");
      }
      return;
    }
    try {
      const saved = JSON.parse(raw);
      if (saved?.employee_id) setUser(saved);
    } catch {
      // ignore malformed data
    }
  }, []);

  // Persist user in sessionStorage so admin page can read it
  useEffect(() => {
    if (user) {
      sessionStorage.setItem("finsolve_user", JSON.stringify(user));
      loadSessions();
    }
  }, [user]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginEmail.trim()) return;
    setIsLoggingIn(true);
    setLoginError("");
    try {
      const data = await loginEmployee(loginEmail.trim());
      setUser(data);
    } catch (e: any) {
      setLoginError(e.message || "incorrect email id, please try again.");
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem("finsolve_user");
    setUser(null);
    setSessions([]);
    setMessages([]);
    setCurrentSessionId(null);
    setCurrentSessionTitle("");
  };

  const loadSessions = async () => {
    if (!user) return;
    try {
      const data = await fetchHistory(user.employee_id);
      setSessions(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadSessionChat = async (session: Session) => {
    if (!user) return;
    try {
      const data = await fetchSessionHistory(session.session_id, user.employee_id);
      const mapped: Message[] = data.flatMap((row) => [
        { role: "user" as const, content: row.query, timestamp: row.timestamp },
        { role: "assistant" as const, content: row.response, timestamp: row.timestamp },
      ]);
      setMessages(mapped);
      setCurrentSessionId(session.session_id);
      setCurrentSessionTitle(session.first_query || "");
    } catch (e) {
      console.error(e);
    }
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setCurrentSessionTitle("");
    setMessages([]);
  };

  const handleSend = async () => {
    if (!inputVal.trim() || !user) return;
    const userMsg = inputVal.trim();
    setInputVal("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsLoading(true);
    try {
      const res = await sendChatMessage(userMsg, user.employee_id, currentSessionId ?? undefined);
      if (!currentSessionId) {
        setCurrentSessionId(res.session_id);
        setCurrentSessionTitle(userMsg);
        loadSessions();
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.response,
          guardrail_type: res.guardrail_type,
          is_blocked: res.is_blocked,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Could not connect to the API. Please check that the backend is running." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const accessibleCollections = user ? getAccessibleCollections(user.access_role) : [];

  // ── Login screen ────────────────────────────────────────────────────────────

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-50 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 to-blue-50 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-500/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="w-full max-w-md relative z-10 px-4">
          <div className="bg-white rounded-3xl shadow-2xl border border-zinc-100 p-8">
            <div className="flex flex-col items-center mb-8">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-blue-600 flex items-center justify-center text-white text-2xl shadow-xl shadow-indigo-500/30 mb-4 font-bold">
                f
              </div>
              <h1 className="text-2xl font-bold text-zinc-900">FinSolve AI</h1>
              <p className="text-sm text-zinc-500 mt-2 text-center leading-relaxed">
                Enterprise knowledge assistant with role-based access control.
                Enter your work email to sign in.
              </p>
            </div>

            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-sm font-semibold text-zinc-700">Work Email</label>
                <Input
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                  placeholder="e.g. pavan.krishnan@finsolve.com"
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
                {isLoggingIn ? "Authenticating…" : (
                  <><KeyRound className="w-4 h-4" /> Secure Sign-In</>
                )}
              </Button>

              {/* Admin portal shortcut */}
              <div className="relative flex items-center gap-3 pt-1">
                <div className="flex-1 h-px bg-zinc-100" />
                <span className="text-[11px] text-zinc-400 font-medium">or</span>
                <div className="flex-1 h-px bg-zinc-100" />
              </div>
              <button
                type="button"
                onClick={() => router.push("/admin")}
                className="w-full h-11 flex items-center justify-center gap-2 rounded-xl border border-indigo-200 bg-indigo-50 text-indigo-700 text-sm font-medium hover:bg-indigo-100 transition-all active:scale-[0.98]"
              >
                <Shield className="w-4 h-4" />
                Go to Admin Portal
              </button>
            </form>

            <p className="text-center text-[10px] text-zinc-400 mt-5 leading-relaxed">
              Admin access is restricted to authorised personnel.<br />Sign in with your admin email to unlock the Admin Panel.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Main app ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen bg-[#f8f9fb] text-zinc-900 font-sans overflow-hidden">

      {/* ── Sidebar ────────────────────────────────────────────────────────────── */}
      <div className="w-72 bg-white border-r flex flex-col shadow-sm flex-shrink-0">

        {/* Logo */}
        <div className="p-5 border-b">
          <h1 className="text-xl font-bold flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">
            <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-600 to-blue-600 flex items-center justify-center text-white text-sm shadow-md shrink-0">f</span>
            FinSolve AI
          </h1>
        </div>

        {/* User card */}
        <div className="px-4 pt-4 pb-2">
          <div className="p-3 bg-zinc-50 border border-zinc-200 rounded-xl mb-3">
            <div className="flex items-center gap-3 mb-2">
              <Avatar className="w-9 h-9 border bg-white shrink-0">
                <AvatarFallback className="bg-indigo-100 text-indigo-700 font-bold text-xs">
                  {getInitials(user.name)}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-bold text-zinc-900 truncate">{user.name}</h3>
                <p className="text-[10px] text-zinc-500 truncate">{user.email}</p>
              </div>
            </div>

            {/* Role badge */}
            <div className="flex items-center gap-1.5 mb-2">
              <span className={`text-[10px] px-2 py-0.5 rounded-full border font-semibold ${ROLE_BADGE[user.access_role] ?? ROLE_BADGE["General"]}`}>
                {user.access_role}
              </span>
              <span className="text-[10px] text-zinc-400">{user.department}</span>
            </div>

            {/* Collection access */}
            <div>
              <p className="text-[9px] text-zinc-400 uppercase tracking-wider font-semibold mb-1">Collection Access</p>
              <div className="flex flex-wrap gap-1">
                {accessibleCollections.map((col) => (
                  <CollectionBadge key={col} collection={col} />
                ))}
              </div>
            </div>
          </div>

          {/* New chat button */}
          <Button
            onClick={handleNewChat}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm border-0 justify-start gap-2 h-9"
          >
            <PlusCircle className="w-4 h-4" /> New Chat
          </Button>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-3 py-2">
          <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider px-2 mb-2">
            Recent Chats
          </p>
          {sessions.length === 0 ? (
            <p className="text-xs text-zinc-400 px-2">No previous chats yet.</p>
          ) : (
            <div className="flex flex-col gap-0.5">
              {sessions.map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => loadSessionChat(s)}
                  className={`flex items-center gap-2 text-left px-2.5 py-2 text-sm rounded-lg transition-colors w-full group ${
                    currentSessionId === s.session_id
                      ? "bg-indigo-50 text-indigo-700 font-medium"
                      : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5 opacity-60 shrink-0" />
                  <span className="truncate flex-1 text-xs leading-relaxed">
                    {truncate(s.first_query, 35)}
                  </span>
                  <ChevronRight className="w-3 h-3 opacity-0 group-hover:opacity-40 shrink-0 transition-opacity" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Sidebar footer — admin link + logout */}
        <div className="px-3 pb-4 pt-2 border-t space-y-1">
          {user.is_admin && (
            <button
              onClick={() => router.push("/admin")}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-indigo-700 hover:bg-indigo-50 rounded-lg transition-colors"
            >
              <Shield className="w-4 h-4" />
              Admin Panel
              <span className="ml-auto text-[9px] px-1.5 py-0.5 bg-indigo-100 text-indigo-600 rounded uppercase tracking-wide font-bold">
                Admin
              </span>
            </button>
          )}
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-zinc-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </div>

      {/* ── Main area ──────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Header */}
        <header className="h-14 border-b bg-white/90 backdrop-blur-md flex items-center px-6 flex-shrink-0 shadow-sm">
          <div className="flex items-center gap-2 min-w-0">
            <Database className="w-4 h-4 text-indigo-400 shrink-0" />
            <h2 className="text-sm font-semibold text-zinc-800 truncate">
              {currentSessionId
                ? truncate(currentSessionTitle || "Chat", 60)
                : "New Knowledge Base Query"}
            </h2>
          </div>
        </header>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="flex flex-col items-center text-zinc-400 gap-4 max-w-sm text-center animate-in fade-in zoom-in duration-500">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-50 to-blue-50 border border-indigo-100 flex items-center justify-center">
                  <Info className="w-8 h-8 text-indigo-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-zinc-700 mb-1">
                    Hi, {user.name.split(" ")[0]}! 👋
                  </h3>
                  <p className="text-sm leading-relaxed">
                    Ask about company policies, your domain docs, or anything within your{" "}
                    <strong className="text-zinc-600">{user.access_role}</strong> access.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1 justify-center">
                  {accessibleCollections.map((col) => (
                    <CollectionBadge key={col} collection={col} />
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col pb-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex w-full py-6 px-8 ${
                    msg.role === "assistant" ? "bg-white/70 border-y border-zinc-100" : ""
                  }`}
                >
                  <div className="max-w-3xl mx-auto w-full flex gap-4">
                    {/* Avatar */}
                    <Avatar className={`w-8 h-8 shrink-0 shadow-sm mt-0.5 ${
                      msg.role === "user"
                        ? "bg-zinc-700"
                        : "bg-gradient-to-br from-indigo-500 to-blue-600 p-[2px]"
                    }`}>
                      {msg.role === "assistant" ? (
                        <div className="w-full h-full bg-white rounded-full flex items-center justify-center text-indigo-600 font-bold text-xs">f</div>
                      ) : (
                        <AvatarFallback className="bg-zinc-800 text-zinc-100 text-xs">
                          {getInitials(user.name)}
                        </AvatarFallback>
                      )}
                    </Avatar>

                    {/* Content */}
                    <div className="flex-1 min-w-0 pt-0.5">
                      {/* Guardrail banner (assistant messages only) */}
                      {msg.role === "assistant" && msg.guardrail_type && (
                        <GuardrailBanner type={msg.guardrail_type} isBlocked={!!msg.is_blocked} />
                      )}

                      {/* Message body */}
                      {msg.role === "assistant" ? (
                        <div className={`prose prose-zinc prose-sm max-w-none text-zinc-800 leading-relaxed break-words ${
                          msg.is_blocked ? "opacity-75" : ""
                        }`}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="text-sm text-zinc-800 whitespace-pre-wrap leading-relaxed">
                          {msg.content}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="flex w-full py-6 px-8 bg-white/70 border-y border-zinc-100 animate-in fade-in duration-300">
                  <div className="max-w-3xl mx-auto w-full flex gap-4">
                    <Avatar className="w-8 h-8 shrink-0 bg-gradient-to-br from-indigo-500 to-blue-600 p-[2px] shadow-sm">
                      <div className="w-full h-full bg-white rounded-full flex items-center justify-center text-indigo-600 font-bold text-xs">f</div>
                    </Avatar>
                    <div className="flex items-center gap-1.5 pt-2">
                      {[0, 150, 300].map((delay) => (
                        <span
                          key={delay}
                          className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce"
                          style={{ animationDelay: `${delay}ms` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-5 bg-white/80 backdrop-blur-md border-t border-zinc-200/60 flex-shrink-0">
          <div className="max-w-3xl mx-auto flex items-center gap-3 bg-white p-2 rounded-2xl shadow-sm border border-zinc-200 focus-within:ring-2 focus-within:ring-indigo-500/20 focus-within:border-indigo-300 transition-all">
            <Input
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={`Ask FinSolve AI as ${user.name.split(" ")[0]}…`}
              className="flex-1 border-0 shadow-none focus-visible:ring-0 px-4 bg-transparent h-10 text-zinc-700"
              disabled={isLoading}
            />
            <Button
              onClick={handleSend}
              disabled={isLoading || !inputVal.trim()}
              className="w-10 h-10 shrink-0 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-600/20 transition-all active:scale-95"
              size="icon"
            >
              <Send className="w-4 h-4 ml-0.5" />
            </Button>
          </div>
          <p className="text-center text-[10px] text-zinc-400 mt-2">
            Responses are grounded in your authorized documents. Always verify critical information with source files.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense>
      <FinSolveDashboard />
    </Suspense>
  );
}
