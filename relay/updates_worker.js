// Fintan relay at updates.fintanlabs.com: the code it runs, published so anyone can check that it keeps nothing.
// The only address the front door may contact (door/net.py ALLOWED_HOST).
//
//   GET  /latest                       the newest update's manifest (same answer for everyone)
//   GET  /update/<date>/<file>         a file of a signed update (same answer for everyone)
//   POST /topic   {"q": [5 topics]}    for each topic, encyclopedia articles whose titles name it (the client sends
//                                      its topic among 4 decoys; this Worker answers all five and cannot tell which)
//
// KEEPS NOTHING: no console.log, no storage written, no analytics, no cookies, no request headers or addresses
// read or echoed. Worker logs are switched off in wrangler.toml ([observability] enabled = false).
// What this file cannot change: Cloudflare's network sees the caller's IP address to deliver the answer.

const MAX_TOPICS = 6;
const MAX_TOPIC_CHARS = 80;
const ARTICLES_PER_TOPIC = 3;
const WIKI = "https://en.wikipedia.org/w/api.php";
const WIKI_UA = "FintanUpdates/0.1 (https://fintanlabs.com)";   // Wikipedia's API asks every tool to name itself
const STOP = new Set("the a an of in on at to for and or by with from about".split(" "));

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function words(s) {
  return s.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w && !STOP.has(w));
}

async function wiki(params) {
  const url = WIKI + "?" + new URLSearchParams({ format: "json", formatversion: "2", ...params });
  const r = await fetch(url, { headers: { "User-Agent": WIKI_UA } });
  if (!r.ok) return null;
  return r.json();
}

// and lists, and missed the main article ("Black hole")
function stem(w) {
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && /(?:ches|shes|sses|xes|zes)$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !/(?:ss|us|is)$/.test(w)) return w.slice(0, -1);
  return w;
}

function stems(s) {
  return words(s).map(stem);
}

async function page(title) {
  const e = await wiki({ action: "query", prop: "extracts", explaintext: "1", titles: title, redirects: "1" });
  const p = e && e.query && e.query.pages && e.query.pages[0];
  if (!p || p.missing || !p.extract || /may (?:also )?refer to/.test(p.extract.slice(0, 400))) return null;
  return { title: p.title, url: "https://en.wikipedia.org/wiki/" + encodeURIComponent(p.title.replace(/ /g, "_")),
           text: p.extract.slice(0, 200000) };
}

async function articles(topic) {
  const want = stems(topic);
  if (!want.length) return [];
  const out = [];
  const seen = new Set();
  // 1. the topic's own article: the page with that name, or the one its name leads to ("Black holes" -> "Black hole")
  const own = await page(topic);
  if (own) { out.push(own); seen.add(own.title); }
  // 2. search results whose title ends with the topic, so the topic is what the article is about ("List of black
  //    holes", "Supermassive black hole"; not "Black Holes and Revelations", not "Black Hole (film)"); if none, the
  //    old rule: titles that name every word of the topic
  const s = await wiki({ action: "query", list: "search", srsearch: topic, srlimit: "10", srnamespace: "0" });
  const found = ((s && s.query && s.query.search) || []).map((x) => x.title).filter((t) => !seen.has(t));
  const endsWith = (t) => { const ts = stems(t); return ts.length >= want.length && want.every((w, i) => ts[ts.length - want.length + i] === w); };
  let pick = found.filter(endsWith);
  if (!pick.length && !out.length) pick = found.filter((t) => { const ts = new Set(stems(t)); return want.every((w) => ts.has(w)); });
  for (const t of pick) {
    if (out.length >= ARTICLES_PER_TOPIC) break;
    const p = await page(t);
    if (p && !seen.has(p.title)) { out.push(p); seen.add(p.title); }
  }
  return out;
}

// stored as parts "<key>#0", "<key>#1", ... with "<key>" = {"chunks": n, "size": bytes}, and streamed back in order.
async function storedFile(env, ctx, key, type) {
  const meta = await env.UPDATES.get(key, { type: "json" });
  if (!meta || !meta.chunks) return null;
  const { readable, writable } = new TransformStream();
  const pump = (async () => {
    const w = writable.getWriter();
    try {
      for (let i = 0; i < meta.chunks; i++) {
        const part = await env.UPDATES.get(key + "#" + i, { type: "arrayBuffer" });
        if (!part) throw new Error("missing part");
        await w.write(new Uint8Array(part));
      }
      await w.close();
    } catch (e) {
      await w.abort(e);
    }
  })();
  ctx.waitUntil(pump);
  return new Response(readable, { headers: { "Content-Type": type, "Content-Length": String(meta.size), "Cache-Control": "public, max-age=86400" } });
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (request.method === "GET" && path === "/latest") {
      const text = await env.UPDATES.get("latest.json");
      return text ? new Response(text, { headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=600" } })
                  : json({ error: "no update yet" }, 404);
    }
    const m = path.match(/^\/update\/(\d{4}-\d{2}-\d{2})\/([A-Za-z0-9_.-]+)$/);
    if (request.method === "GET" && m) {
      const r = await storedFile(env, ctx, m[1] + "/" + m[2], "application/octet-stream");
      return r || json({ error: "no such file" }, 404);
    }
    if (request.method === "POST" && path === "/topic") {
      let body;
      try { body = await request.json(); } catch (e) { return json({ error: "send {\"q\": [topics]}" }, 400); }
      const q = Array.isArray(body && body.q) ? body.q : [];
      if (!q.length || q.length > MAX_TOPICS || q.some((t) => typeof t !== "string" || !t.trim() || t.length > MAX_TOPIC_CHARS)) {
        return json({ error: "1 to 6 topics of at most 80 characters" }, 400);
      }
      const results = {};
      for (const t of q) results[t] = await articles(t);
      return json({ results });
    }
    return json({ error: "not found" }, 404);
  },
};
