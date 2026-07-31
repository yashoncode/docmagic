"use client";

import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { ArrowUp, MessagesSquare, ThumbsDown, ThumbsUp } from "lucide-react";
import { sendFeedback, type Msg } from "@/lib/api";

type Props = {
  messages: Msg[];
  status: string | null;
  busy: boolean;
  suggestions: string[];
  model: string;
  onSend: (question: string) => void;
};

export default function Chat({
  messages, status, busy, suggestions, model, onSend,
}: Props) {
  const [draft, setDraft] = useState("");
  const [rated, setRated] = useState<Record<number, "up" | "down">>({});
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = draft.trim();
    if (!q || busy) return;
    setDraft("");
    onSend(q);
  }

  function rate(i: number, rating: "up" | "down") {
    setRated((r) => ({ ...r, [i]: rating }));
    sendFeedback({
      rating,
      question: messages[i - 1]?.content ?? "",
      answer: messages[i]?.content ?? "",
      model,
    });
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-5">
          {messages.length === 0 && (
            <div className="py-6 text-center">
              <div className="mx-auto grid size-11 place-items-center rounded-full bg-surface ring-1 ring-border">
                <MessagesSquare className="size-[18px] text-muted" aria-hidden />
              </div>
              <p className="display mt-3 text-[15px]">Ask anything</p>
              <p className="mt-1 text-xs text-muted">
                Every answer cites the file, page or sheet it came from.
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => onSend(s)}
                    className="btn btn-ghost rounded-full py-1.5 text-xs"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-surface px-3.5 py-2 text-[13px] whitespace-pre-wrap">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={i}>
                <div className="md text-[13.5px]">
                  <Markdown>{m.content}</Markdown>
                </div>
                {m.content && !(busy && i === messages.length - 1) && (
                  <div className="mt-2 flex gap-0.5">
                    {(["up", "down"] as const).map((r) => {
                      const Icon = r === "up" ? ThumbsUp : ThumbsDown;
                      return (
                        <button
                          key={r}
                          type="button"
                          onClick={() => rate(i, r)}
                          aria-label={r === "up" ? "Good answer" : "Bad answer"}
                          aria-pressed={rated[i] === r}
                          className={`rounded-md p-1.5 hover:bg-surface ${
                            rated[i] === r ? "text-accent" : "text-muted"
                          }`}
                        >
                          <Icon className="size-3.5" aria-hidden />
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            ),
          )}

          {status && (
            <div className="flex items-center gap-2" aria-live="polite">
              <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-accent" />
              <span className="meta pulse text-muted">{status}</span>
            </div>
          )}
          <div ref={bottom} />
        </div>
      </div>

      <form onSubmit={submit} className="shrink-0 border-t border-border p-3 sm:px-6">
        <div className="mx-auto flex w-full max-w-2xl items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) submit(e);
            }}
            rows={1}
            placeholder="Ask about your documents…"
            aria-label="Ask about your documents"
            className="field max-h-40 min-h-[40px] flex-1 resize-y rounded-xl"
          />
          <button
            type="submit"
            disabled={busy || !draft.trim()}
            aria-label="Send"
            className="btn btn-primary size-10 shrink-0 rounded-xl p-0"
          >
            <ArrowUp size={20} aria-hidden />
          </button>
        </div>
      </form>
    </div>
  );
}
