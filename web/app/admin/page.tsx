"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertCircle, ArrowLeft, Check, Loader2 } from "lucide-react";
import { listUsers, setUserTokens, type User } from "@/lib/api";

/** Admin console: who signed up, what they've burned, and their credit balance.
 *  Gated server-side by ADMIN_EMAILS — this page just reports what the API allows. */
export default function AdminPage() {
  const [users, setUsers] = useState<User[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Record<number, string>>({});
  const [saving, setSaving] = useState<number | null>(null);

  useEffect(() => {
    listUsers()
      .then(setUsers)
      .catch((e) => setError(e.message));
  }, []);

  async function save(u: User) {
    const tokens = Number(draft[u.id]);
    if (!Number.isFinite(tokens)) return;
    setSaving(u.id);
    try {
      const updated = await setUserTokens(u.id, Math.trunc(tokens));
      setUsers((list) => (list ?? []).map((x) => (x.id === u.id ? { ...x, ...updated } : x)));
      setDraft((d) => {
        const next = { ...d };
        delete next[u.id];
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not update credits.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <main className="mx-auto max-w-4xl space-y-4 px-4 py-6 sm:px-6">
      <div className="rise flex items-end gap-4">
        <div>
          <p className="eyebrow text-accent">Admin</p>
          <h1 className="display mt-1.5 text-[26px]">Users and credits</h1>
        </div>
        <Link href="/" className="btn btn-ghost ml-auto">
          <ArrowLeft className="size-4" aria-hidden />
          Back
        </Link>
      </div>

      {error && (
        <p className="well flex items-start gap-2 px-3 py-2.5 text-[13px]">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
          {error}
        </p>
      )}

      <section className="panel overflow-hidden">
        {!users && !error && <p className="meta pulse p-4 text-muted">loading…</p>}
        {users?.length === 0 && <p className="p-4 text-sm text-muted">Nobody has signed in yet.</p>}
        {!!users?.length && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[13px]">
              <thead className="sticky-head">
                <tr className="eyebrow text-muted">
                  <th className="px-4 py-2.5">User</th>
                  <th className="px-4 py-2.5 text-right">Used</th>
                  <th className="px-4 py-2.5 text-right">Left</th>
                  <th className="px-4 py-2.5">Set credits</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className="border-t border-border">
                    <td className="px-4 py-2.5">
                      <span className="font-medium">{u.name}</span>
                      {u.is_admin && <span className="chip ml-2">admin</span>}
                      <span className="meta block text-muted">{u.email}</span>
                    </td>
                    <td className="px-4 py-2.5 text-right">{u.tokens_used.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-right font-medium">
                      {u.tokens_left.toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5">
                      <form
                        onSubmit={(e) => {
                          e.preventDefault();
                          void save(u);
                        }}
                        className="flex items-center gap-1.5"
                      >
                        <label className="sr-only" htmlFor={`tokens-${u.id}`}>
                          Token credits for {u.email}
                        </label>
                        <input
                          id={`tokens-${u.id}`}
                          type="number"
                          min={0}
                          step={1000}
                          value={draft[u.id] ?? String(u.tokens_left)}
                          onChange={(e) => setDraft((d) => ({ ...d, [u.id]: e.target.value }))}
                          className="field meta w-28"
                        />
                        <button
                          type="submit"
                          disabled={saving === u.id || draft[u.id] === undefined}
                          aria-label={`Save credits for ${u.email}`}
                          className="btn btn-ghost size-9 p-0"
                        >
                          {saving === u.id ? (
                            <Loader2 className="size-4 animate-spin" aria-hidden />
                          ) : (
                            <Check className="size-4" aria-hidden />
                          )}
                        </button>
                      </form>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
