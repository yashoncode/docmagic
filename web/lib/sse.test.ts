/** Check the SSE frame parser. Node 24 strips types natively:  node lib/sse.test.ts
 *
 * The parser is the one piece of pure logic in the frontend that can break
 * silently — frames arrive split across arbitrary chunk boundaries, sse-starlette
 * uses CRLF, and keep-alive comments must be ignored, not parsed.
 */
import assert from "node:assert/strict";
import { sse } from "./api.ts";

/** A Response whose body yields exactly these chunks, to force split boundaries. */
function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const c of chunks) controller.enqueue(encoder.encode(c));
        controller.close();
      },
    }),
  );
}

async function collect(chunks: string[]) {
  const out: { event: string; data: unknown }[] = [];
  for await (const ev of sse(streamOf(chunks))) out.push(ev);
  return out;
}

const events = await collect([
  // CRLF, as sse-starlette emits
  'event: status\r\ndata: {"route": "documents"}\r\n\r\n',
  // a frame split mid-JSON across two network chunks
  'event: token\r\ndata: {"text": "Hel',
  'lo"}\r\n\r\n',
  // keep-alive comment between frames must be ignored
  ": ping\r\n\r\n",
  // bare LF must work too
  'event: token\ndata: {"text": " world"}\n\n',
  // two frames arriving in one chunk
  'event: token\r\ndata: {"text": "!"}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n',
]);

assert.deepEqual(
  events.map((e) => e.event),
  ["status", "token", "token", "token", "done"],
  "every frame is yielded exactly once, comments excluded",
);
assert.deepEqual(events[0].data, { route: "documents" });
assert.equal(
  events
    .filter((e) => e.event === "token")
    .map((e) => (e.data as { text: string }).text)
    .join(""),
  "Hello world!",
  "text survives chunk boundaries",
);

// a truncated trailing frame must be dropped, not throw
assert.deepEqual(await collect(['event: token\r\ndata: {"text": "x"']), []);

console.log("ok");
