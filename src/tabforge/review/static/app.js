// TabForge review UI — vanilla JS, no build step.
// Draws the pitch contour with note boundaries (the key debugging view),
// synced audio playback with a moving cursor, loop-at-speed, and click-to-edit
// note correction (POSTed to /api/correct, which appends to corrections.json).

const ART = ["pluck", "hammer_on", "pull_off", "slide_in", "slide_out", "bend", "release", "vibrato"];
let DATA = null;
let AUDIO = new Audio("/api/audio");
let sel = null;         // {t0, t1} loop selection in seconds
let dragging = null;

const cv = document.getElementById("contour");
const ctx = cv.getContext("2d");

function fit() {
  cv.width = cv.clientWidth * devicePixelRatio;
  cv.height = 320 * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
window.addEventListener("resize", () => { fit(); draw(); });

async function load() {
  const r = await fetch("/api/song");
  DATA = await r.json();
  document.getElementById("meta").textContent =
    `song ${DATA.song_id} · tempo ${Math.round(DATA.tempo_bpm)} bpm · ${DATA.notes.length} notes`;
  document.getElementById("tab").textContent = DATA.tab;
  document.getElementById("notecount").textContent = `(${DATA.notes.length})`;
  buildTable();
  fit(); draw();
}

// ---- coordinate helpers -------------------------------------------------
function bounds() {
  const times = DATA.times;
  const tMax = times.length ? times[times.length - 1] : 1;
  const cents = DATA.hz.map(hz => hz > 0 ? 1200 * Math.log2(hz / 440) + 6900 : NaN);
  let lo = Infinity, hi = -Infinity;
  for (const c of cents) { if (!isNaN(c)) { lo = Math.min(lo, c); hi = Math.max(hi, c); } }
  if (!isFinite(lo)) { lo = 4000; hi = 8000; }
  lo -= 100; hi += 100;
  return { tMax, lo, hi, cents };
}
function xOf(t, tMax, w) { return (t / tMax) * (w - 60) + 50; }
function yOf(c, lo, hi, h) { return h - 30 - ((c - lo) / (hi - lo)) * (h - 50); }

// ---- drawing ------------------------------------------------------------
function draw() {
  if (!DATA) return;
  const w = cv.clientWidth, h = 320;
  ctx.clearRect(0, 0, w, h);
  const { tMax, lo, hi, cents } = bounds();

  // onsets
  ctx.strokeStyle = "rgba(245,158,11,.5)";
  for (const o of DATA.onsets) {
    const x = xOf(o.time_s, tMax, w);
    ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, h - 30); ctx.stroke();
  }

  // note boundaries + nominal pitch lines
  for (const n of DATA.notes) {
    const x0 = xOf(n.start_s, tMax, w);
    ctx.strokeStyle = "rgba(255,107,107,.7)";
    ctx.beginPath(); ctx.moveTo(x0, 20); ctx.lineTo(x0, h - 30); ctx.stroke();
    const nominal = n.midi * 100 + (n.cents_offset || 0);
    const y = yOf(nominal, lo, hi, h);
    ctx.strokeStyle = "rgba(74,222,128,.9)"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xOf(n.start_s, tMax, w), y); ctx.lineTo(xOf(n.end_s, tMax, w), y);
    ctx.stroke(); ctx.lineWidth = 1;
  }

  // contour
  ctx.strokeStyle = "#4aa3ff"; ctx.beginPath();
  let pen = false;
  for (let i = 0; i < DATA.times.length; i++) {
    const c = cents[i];
    if (isNaN(c)) { pen = false; continue; }
    const x = xOf(DATA.times[i], tMax, w), y = yOf(c, lo, hi, h);
    if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // selection
  if (sel) {
    ctx.fillStyle = "rgba(124,92,255,.18)";
    const x0 = xOf(sel.t0, tMax, w), x1 = xOf(sel.t1, tMax, w);
    ctx.fillRect(x0, 20, x1 - x0, h - 50);
  }

  // playback cursor
  const x = xOf(AUDIO.currentTime, tMax, w);
  ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, h - 30); ctx.stroke();
}

function tick() { draw(); requestAnimationFrame(tick); }

// ---- interaction --------------------------------------------------------
function timeAt(evt) {
  const rect = cv.getBoundingClientRect();
  const w = cv.clientWidth;
  const { tMax } = bounds();
  const px = evt.clientX - rect.left;
  return Math.max(0, ((px - 50) / (w - 60)) * tMax);
}
cv.addEventListener("mousedown", e => { dragging = timeAt(e); sel = null; });
cv.addEventListener("mousemove", e => {
  if (dragging == null) return;
  const t = timeAt(e);
  sel = { t0: Math.min(dragging, t), t1: Math.max(dragging, t) };
  document.getElementById("selinfo").textContent =
    `loop ${sel.t0.toFixed(2)}s – ${sel.t1.toFixed(2)}s`;
  draw();
});
cv.addEventListener("mouseup", e => {
  const t = timeAt(e);
  if (sel && Math.abs(sel.t1 - sel.t0) < 0.02) {  // a click, not a drag -> select note
    sel = null;
    const idx = DATA.notes.findIndex(n => t >= n.start_s && t <= n.end_s);
    if (idx >= 0) highlightRow(idx);
  } else if (!sel) {
    AUDIO.currentTime = t;
  }
  dragging = null;
});

// ---- transport ----------------------------------------------------------
document.getElementById("play").onclick = () => {
  AUDIO.playbackRate = parseFloat(document.getElementById("speed").value);
  if (sel) AUDIO.currentTime = sel.t0;
  AUDIO.play();
};
document.getElementById("stop").onclick = () => { AUDIO.pause(); };
document.getElementById("speed").onchange = () =>
  AUDIO.playbackRate = parseFloat(document.getElementById("speed").value);
AUDIO.addEventListener("timeupdate", () => {
  if (document.getElementById("loop").checked && sel && AUDIO.currentTime >= sel.t1) {
    AUDIO.currentTime = sel.t0;
  }
});

// ---- editable notes table ----------------------------------------------
function buildTable() {
  const tb = document.querySelector("#notes tbody");
  tb.innerHTML = "";
  DATA.notes.forEach((n, i) => {
    const tr = document.createElement("tr");
    tr.dataset.i = i;
    tr.innerHTML = `
      <td>${i}</td>
      <td>${n.start_s.toFixed(3)}</td>
      <td><input class="narrow" type="number" value="${n.midi}" data-k="midi"></td>
      <td><input class="narrow" type="number" value="${n.string ?? ''}" data-k="string"></td>
      <td><input class="narrow" type="number" value="${n.fret ?? ''}" data-k="fret"></td>
      <td>${artSelect(n.entry, "entry")}</td>
      <td>${artSelect(n.exit, "exit", true)}</td>
      <td>${n.vibrato_hz ? '~' : ''}</td>
      <td><button class="save" data-i="${i}">save</button></td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll("button.save").forEach(b =>
    b.onclick = () => saveRow(parseInt(b.dataset.i)));
  tb.querySelectorAll("tr").forEach(tr =>
    tr.onclick = () => highlightRow(parseInt(tr.dataset.i), false));
}
function artSelect(val, key, allowNone) {
  const opts = (allowNone ? [""] : []).concat(ART)
    .map(a => `<option value="${a}" ${a === (val || "") ? "selected" : ""}>${a || "—"}</option>`)
    .join("");
  return `<select data-k="${key}">${opts}</select>`;
}
async function saveRow(i) {
  const tr = document.querySelector(`#notes tbody tr[data-i="${i}"]`);
  const payload = { note_index: i };
  tr.querySelectorAll("[data-k]").forEach(el => {
    const k = el.dataset.k;
    let v = el.value;
    if (v === "") { payload[k] = k === "exit" ? null : undefined; return; }
    payload[k] = (k === "midi" || k === "string" || k === "fret") ? parseInt(v) : v;
  });
  const r = await fetch("/api/correct", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const res = await r.json();
  DATA.notes = res.notes;
  DATA.tab = res.tab;
  document.getElementById("tab").textContent = res.tab;
  buildTable(); draw();
  highlightRow(i, false);
}
function highlightRow(i, scroll = true) {
  document.querySelectorAll("#notes tbody tr").forEach(tr =>
    tr.classList.toggle("active", parseInt(tr.dataset.i) === i));
  const n = DATA.notes[i];
  if (n) sel = null;
  if (scroll) {
    const tr = document.querySelector(`#notes tbody tr[data-i="${i}"]`);
    if (tr) tr.scrollIntoView({ block: "nearest" });
  }
  draw();
}

load().then(() => tick());
