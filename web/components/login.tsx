"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  FileSpreadsheet,
  Loader2,
  MessageSquareQuote,
  Scale,
  Sparkles,
} from "lucide-react";
import { ThemeToggle } from "@/components/topbar";
import { emailAuth, googleLogin, type Config, type User } from "@/lib/api";

/** Google Identity Services, loaded from its own CDN — the only part of the API we call. */
declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (o: { client_id: string; callback: (r: { credential: string }) => void }) => void;
          renderButton: (el: HTMLElement, o: Record<string, string>) => void;
        };
      };
    };
  }
}

const GSI = "https://accounts.google.com/gsi/client";

/** Loads the GSI script once per page and resolves when `window.google` exists. */
function loadGsi(): Promise<void> {
  if (window.google) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI}"]`);
    const el = existing ?? Object.assign(document.createElement("script"), { src: GSI, async: true });
    el.addEventListener("load", () => resolve());
    el.addEventListener("error", () => reject(new Error("Couldn't reach Google sign-in.")));
    if (!existing) document.head.appendChild(el);
  });
}

const PERKS = [
  {
    icon: FileSpreadsheet,
    title: "Drop in the messy stuff",
    text: "Rate cards, SOPs, contracts, ERP exports — scanned PDFs included.",
  },
  {
    icon: MessageSquareQuote,
    title: "Answers you can check",
    text: "Replies cite the page or sheet they came from, so nothing is taken on trust.",
  },
  {
    icon: Scale,
    title: "Billing that adds up",
    text: "Check a bill against the agreed rates — rate x quantity is computed, not guessed.",
  },
  {
    icon: BarChart3,
    title: "Charts from a sentence",
    text: "Describe the view you want and it is built from your own rows.",
  },
];

/** The mark: initials in a gradient tile. Cheaper than an image and theme-proof. */
function Mark() {
  return (
    <span
      aria-hidden
      className="grid size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-accent to-accent-hover text-background shadow-[var(--shadow)]"
    >
      <Sparkles className="size-[18px]" />
    </span>
  );
}

/** Sign-in gate: the pitch on the left, the form on the right. Google mints an ID token
 *  in the browser and the API verifies it, so no token is ever kept client-side. */
export default function Login({ config, onSignedIn }: { config: Config; onSignedIn: (u: User) => void }) {
  const slot = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [busy, setBusy] = useState(false);
  const signup = mode === "signup";

  useEffect(() => {
    if (!config.googleClientId) return;
    let cancelled = false;
    loadGsi()
      .then(() => {
        if (cancelled || !slot.current || !window.google) return;
        window.google.accounts.id.initialize({
          client_id: config.googleClientId,
          callback: ({ credential }) =>
            googleLogin(credential)
              .then(onSignedIn)
              .catch((e) => setError(e instanceof Error ? e.message : "Sign-in failed.")),
        });
        window.google.accounts.id.renderButton(slot.current, {
          theme: "outline",
          size: "large",
          shape: "pill",
          text: "continue_with",
          logo_alignment: "center",
          // Google sizes the button once, in pixels — match the card so it doesn't
          // sit narrower than the form below it (400 is its own maximum)
          width: String(Math.min(Math.max(slot.current.offsetWidth, 200), 400)),
        });
      })
      .catch((e) => setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [config.googleClientId, onSignedIn]);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      onSignedIn(
        await emailAuth(mode, {
          email: String(form.get("email")),
          password: String(form.get("password")),
          name: String(form.get("name") ?? ""),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed.");
      setBusy(false); // on success the page swaps out, so only failure needs the reset
    }
  }

  return (
    <div className="grid min-h-dvh lg:grid-cols-[1.05fr_minmax(0,0.95fr)]">
      {/* ---- the pitch. Hidden on small screens, where the card carries its own header ---- */}
      <section className="relative hidden flex-col justify-between gap-10 p-10 xl:p-14 lg:flex">
        <div className="rise flex items-center gap-3">
          <Mark />
          <span className="display text-[17px]">DocMagic</span>
          <span className="chip">XEON AI</span>
        </div>

        <div className="max-w-lg">
          <p className="eyebrow rise text-accent">Document intelligence</p>
          <h1 className="display rise mt-3 text-[clamp(2.25rem,4vw,3.25rem)]">
            Ask your documents
            <br />
            anything.
          </h1>
          <p className="rise mt-4 text-[15px] leading-relaxed text-muted">
            A reading room for logistics, ERP and CRM paperwork. Upload it, and get a reviewed
            summary, cited answers and charts built from your own numbers.
          </p>

          <ul className="stagger mt-9 space-y-5">
            {PERKS.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3.5">
                <span
                  aria-hidden
                  className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent"
                >
                  <Icon className="size-4" />
                </span>
                <div>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="mt-0.5 text-[13px] leading-relaxed text-muted">{text}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <p className="eyebrow text-muted">Logistics · Warehousing · ERP · CRM</p>
      </section>

      {/* ---- the form ---- */}
      <section className="flex flex-col justify-center gap-5 border-border p-4 sm:p-8 lg:border-l lg:bg-surface-2">
        <div className="ml-auto lg:-mt-4">
          <ThemeToggle />
        </div>

        <div className="panel pop mx-auto w-full max-w-[27rem] p-6 sm:p-8">
          {/* small screens get the brand here — the pitch column is hidden */}
          <div className="mb-6 flex items-center gap-3 lg:hidden">
            <Mark />
            <span className="display text-[17px]">DocMagic</span>
            <span className="chip">XEON AI</span>
          </div>

          <h2 className="display text-[22px]">{signup ? "Create your account" : "Welcome back"}</h2>
          <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
            {signup ? (
              <>
                Free, and it starts with{" "}
                <strong className="font-medium text-foreground">
                  {config.freeTokens.toLocaleString()}
                </strong>{" "}
                AI tokens on the house.
              </>
            ) : (
              "Sign in to pick up where your documents left off."
            )}
          </p>

          {config.googleClientId && (
            <>
              {/* GIS paints its own button; the wrapper only reserves the width */}
              <div ref={slot} className="mt-6 min-h-[40px] [color-scheme:light]" />
              <div className="my-5 flex items-center gap-3">
                <span className="h-px flex-1 bg-border" />
                <span className="eyebrow text-muted">or with email</span>
                <span className="h-px flex-1 bg-border" />
              </div>
            </>
          )}

          <form onSubmit={submit} className={config.googleClientId ? "space-y-3.5" : "mt-6 space-y-3.5"}>
            {signup && (
              <div>
                <label htmlFor="name" className="meta text-muted">
                  Name
                </label>
                <input
                  id="name"
                  name="name"
                  autoComplete="name"
                  placeholder="Yashwanth"
                  className="field mt-1.5"
                />
              </div>
            )}
            <div>
              <label htmlFor="email" className="meta text-muted">
                Work email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                autoComplete="email"
                placeholder="you@company.com"
                className="field mt-1.5"
              />
            </div>
            <div>
              <label htmlFor="password" className="meta text-muted">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                minLength={config.minPassword}
                autoComplete={signup ? "new-password" : "current-password"}
                placeholder={signup ? `at least ${config.minPassword} characters` : "••••••••"}
                className="field mt-1.5"
              />
            </div>

            <button type="submit" disabled={busy} className="btn btn-primary mt-1 w-full">
              {busy ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <ArrowRight className="size-4" aria-hidden />
              )}
              {signup ? "Create free account" : "Sign in"}
            </button>
          </form>

          {/* aria-live: a screen reader must hear the rejection, not just see it */}
          <div aria-live="polite">
            {error && (
              <p className="well mt-4 flex items-start gap-2 px-3 py-2.5 text-[13px]">
                <AlertCircle className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
                {error}
              </p>
            )}
          </div>

          <p className="mt-5 text-center text-[13px] text-muted">
            {signup ? "Already have an account?" : "First time here?"}{" "}
            <button
              type="button"
              onClick={() => {
                setMode(signup ? "login" : "signup");
                setError(null);
              }}
              className="font-medium text-accent underline-offset-2 hover:underline"
            >
              {signup ? "Sign in" : "Create a free account"}
            </button>
          </p>
        </div>

        <p className="mx-auto max-w-[27rem] px-1 text-xs leading-relaxed text-muted">
          Your account keeps only your name, email and token balance. Uploaded documents belong to
          the session and are erased when you sign out.
        </p>
      </section>
    </div>
  );
}
