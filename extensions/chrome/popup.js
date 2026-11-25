// Teach Fintan: sends the readable text of the current page to the user's own Fintan (POST /fintan/teach).
// Nothing is sent anywhere else, and nothing is sent until the button is pressed.
const $ = (id) => document.getElementById(id);
const say = (t) => { $("msg").textContent = t; };

async function settings() {
  const s = await chrome.storage.local.get({ url: "http://127.0.0.1:8900", key: "" });
  return { url: s.url.replace(/\/+$/, ""), key: s.key };
}

async function call(path, body) {
  const s = await settings();
  const headers = { "Content-Type": "application/json" };
  if (s.key) headers["Authorization"] = "Bearer " + s.key;
  const r = await fetch(s.url + path, { method: body ? "POST" : "GET", headers, body: body ? JSON.stringify(body) : undefined });
  if (r.status === 401) throw new Error("your Fintan needs its API key (Settings)");
  return r.json();
}

// the page's main text: an <article> or <main> if there is one, else the body, without scripts, menus and footers
function pageText() {
  const root = document.querySelector("article") || document.querySelector("main") || document.body;
  const copy = root.cloneNode(true);
  copy.querySelectorAll("script, style, nav, footer, header, aside, form, noscript").forEach((n) => n.remove());
  const text = copy.innerText.replace(/\n{3,}/g, "\n\n").trim();
  return { title: document.title, url: location.href, text: text.slice(0, 60000) };
}

$("teach").onclick = async () => {
  say("Reading the page...");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: pageText });
    if (!result || result.text.length < 40) return say("There is not enough text on this page to learn from.");
    const r = await call("/fintan/teach", { text: result.title + ". " + result.text, source: result.url, kind: $("kind").value });
    if (!r.accepted) return say("Not accepted: " + (r.reason || "unknown reason"));
    say(r.kept_as_note ? "Kept as a note (learning for this kind is off)." :
        "Taught. " + r.waiting + " item(s) wait for the next learning run.");
  } catch (e) {
    say("Could not reach your Fintan: " + e.message);
  }
};

$("learn").onclick = async () => {
  try {
    const r = await call("/fintan/learn", {});
    say(r.started ? "Learning started. On one GPU, Fintan answers again when it ends." : "Not started: " + (r.reason || ""));
  } catch (e) {
    say("Could not reach your Fintan: " + e.message);
  }
};

$("save").onclick = async () => {
  const url = $("url").value.trim() || "http://127.0.0.1:8900";
  if (!/^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(url)) {
    const origin = new URL(url).origin + "/*";
    const ok = await chrome.permissions.request({ origins: [origin] });
    if (!ok) return say("Permission for " + origin + " was not given.");
  }
  await chrome.storage.local.set({ url, key: $("key").value });
  say("Saved.");
};

settings().then((s) => { $("url").value = s.url; $("key").value = s.key; });
