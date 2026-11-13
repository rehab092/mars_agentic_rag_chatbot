const $ = (s) => document.querySelector(s);
const chatEl = $("#chat");
const input = $("#input");
const sendBtn = $("#send");

// ---------- Sessions (list kept per browser) ----------
function loadSessions() {
  try { return JSON.parse(localStorage.getItem("pg_sessions") || "[]"); } catch { return []; }
}
function saveSessions(list) {
  try { localStorage.setItem("pg_sessions", JSON.stringify(list)); } catch {}
}
function newId() {
  return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, "");
}

let sessions = loadSessions();
let current = null;
let busy = false;

function renderSessions() {
  const nav = $("#sessions");
  nav.innerHTML = "";
  for (const s of sessions) {
    const b = document.createElement("button");
    b.className = "session" + (current === s.id ? " active" : "");
    b.innerHTML = `<span></span><i class="del" title="Delete">✕</i>`;
    b.querySelector("span").textContent = s.title;
    b.onclick = (e) => {
      if (e.target.classList.contains("del")) return deleteSession(s.id);
      openSession(s.id);
      $("#sidebar").classList.remove("open");
    };
    nav.appendChild(b);
  }
}

async function deleteSession(id) {
  await fetch(`/api/history/${id}`, { method: "DELETE" });
  sessions = sessions.filter((s) => s.id !== id);
  saveSessions(sessions);
  if (current === id) startNew(); else renderSessions();
}

function startNew() {
  current = null;
  chatEl.innerHTML = "";
  chatEl.appendChild(emptyTemplate);
  $("#chatTitle").textContent = "New conversation";
  renderSessions();
  input.focus();
}

async function openSession(id) {
  current = id;
  renderSessions();
  const s = sessions.find((x) => x.id === id);
  $("#chatTitle").textContent = s ? s.title : "Conversation";
  chatEl.innerHTML = "";
  const res = await fetch(`/api/history/${id}`);
  const messages = await res.json();
  for (const m of messages) {
    if (m.role === "user") addUser(m.content);
    else {
      const view = addBot();
      view.setAnswer(m.content, false);
      if (m.meta && m.meta.sources && m.meta.sources.length) view.setSources(m.meta.sources);
      view.steps.remove();
    }
  }
  scrollDown();
}

// ---------- Rendering ----------
const emptyTemplate = $("#empty");
emptyTemplate.querySelectorAll(".suggestions button").forEach((b) =>
  b.addEventListener("click", () => send(b.textContent))
);

function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

function addUser(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="bubble"></div>`;
  div.querySelector(".bubble").textContent = text;
  chatEl.appendChild(div);
}

function renderMarkdown(text) {
  const html = DOMPurify.sanitize(marked.parse(text));
  // Turn [1] / [2][3] into clickable citation chips.
  return html.replace(/\[(\d{1,2})\]/g, '<button class="cite" data-ref="$1">$1</button>');
}

function addBot() {
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = `
    <div class="avatar bot"></div>
    <div class="body">
      <details class="steps" open><summary>Thinking…</summary><ol></ol></details>
      <div class="answer"></div>
      <div class="src-wrap"></div>
    </div>`;
  chatEl.appendChild(div);

  const steps = div.querySelector(".steps");
  const list = steps.querySelector("ol");
  const answer = div.querySelector(".answer");
  const srcWrap = div.querySelector(".src-wrap");
  const stepRows = {};

  answer.addEventListener("click", (e) => {
    const ref = e.target.closest(".cite");
    if (!ref) return;
    const card = srcWrap.querySelector(`[data-id="${ref.dataset.ref}"]`);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1200);
    }
  });

  return {
    steps,
    step(ev) {
      let li = stepRows[ev.name];
      if (!li) {
        li = document.createElement("li");
        li.innerHTML = `<span class="icon"></span><div><div class="name"></div><div class="detail"></div></div>`;
        list.appendChild(li);
        stepRows[ev.name] = li;
      }
      const warn = /Insufficient|Retry limit/.test(ev.name + ev.detail);
      li.className = ev.status === "running" ? "running" : warn ? "warn" : "done";
      li.querySelector(".name").textContent = ev.name;
      li.querySelector(".detail").textContent = ev.detail || "";
      steps.querySelector("summary").textContent =
        ev.status === "running" ? `${ev.name}…` : "Thinking…";
      scrollDown();
    },
    finishSteps(count) {
      steps.querySelector("summary").textContent = `Reasoning · ${count} steps`;
      steps.open = false;
    },
    setAnswer(text, streaming) {
      answer.innerHTML = renderMarkdown(text);
      answer.classList.toggle("cursor", !!streaming);
    },
    setError(msg) {
      answer.innerHTML = `<div class="error"></div>`;
      answer.firstChild.textContent = "Something went wrong: " + msg;
    },
    setSources(sources) {
      srcWrap.innerHTML = `<div class="sources-title">Sources</div><div class="sources"></div>`;
      const grid = srcWrap.querySelector(".sources");
      for (const s of sources) {
        const pages = s.page_end && s.page_end !== s.page_start ? `pp. ${s.page_start}–${s.page_end}` : `p. ${s.page_start}`;
        const kind = { text: "Text", image_descriptions: "Image", tables: "Table" }[s.type] || s.type;
        const card = document.createElement("div");
        card.className = "source";
        card.dataset.id = s.id;
        card.innerHTML = `
          ${s.image_url ? `<img alt="" loading="lazy">` : ""}
          <div style="min-width:0">
            <div class="doc"><span class="num">${s.id}</span> <span class="dname"></span></div>
            <div class="meta">${kind} · ${pages}${s.section ? " · <span class='sec'></span>" : ""}</div>
            <div class="snippet"></div>
          </div>`;
        card.querySelector(".dname").textContent = s.document;
        if (s.section) card.querySelector(".sec").textContent = s.section;
        card.querySelector(".snippet").textContent = s.text;
        const img = card.querySelector("img");
        if (img) {
          img.src = s.image_url;
          img.onerror = () => img.remove();
          img.onclick = (e) => {
            e.stopPropagation();
            $("#lightbox img").src = s.image_url;
            $("#lightbox").classList.add("show");
          };
        }
        card.onclick = () => card.classList.toggle("open");
        grid.appendChild(card);
      }
    },
  };
}

$("#lightbox").onclick = () => $("#lightbox").classList.remove("show");

// ---------- Sending ----------
async function send(text) {
  text = (text || "").trim();
  if (!text || busy) return;

  if (!current) {
    current = newId();
    sessions.unshift({ id: current, title: text.slice(0, 60) });
    saveSessions(sessions);
    $("#chatTitle").textContent = text.slice(0, 60);
    chatEl.innerHTML = "";
    renderSessions();
  }

  busy = true;
  sendBtn.disabled = true;
  input.value = "";
  autosize();

  addUser(text);
  const view = addBot();
  scrollDown();

  let answer = "";
  let stepCount = 0;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: current, message: text }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const raw of events) {
        if (!raw.startsWith("data: ")) continue;
        const ev = JSON.parse(raw.slice(6));
        if (ev.type === "step") {
          view.step(ev);
          if (ev.status !== "running") stepCount++;
        } else if (ev.type === "sources") {
          view.setSources(ev.sources);
        } else if (ev.type === "token") {
          answer += ev.text;
          view.setAnswer(answer, true);
          scrollDown();
        } else if (ev.type === "done") {
          view.setAnswer(answer, false);
          view.finishSteps(stepCount);
        } else if (ev.type === "error") {
          view.setError(ev.message);
        }
      }
    }
  } catch (err) {
    view.setError(err.message);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
    scrollDown();
  }
}

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); }
});
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; }
input.addEventListener("input", autosize);

$("#newChat").onclick = startNew;
$("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");
$("#showSteps").onchange = (e) => document.body.classList.toggle("hide-steps", !e.target.checked);

// Health / status
fetch("/api/health").then((r) => r.json()).then((h) => {
  $("#status").innerHTML = `<span class="dot"></span>${h.chunks} passages indexed · ${h.model}`;
}).catch(() => { $("#status").textContent = "Server unreachable"; });

renderSessions();
input.focus();
