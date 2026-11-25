// Operator UI. While this client controls an active robot it streams the
// gamepad (or keyboard) state at 20 Hz as end-effector velocities; the edge
// holds the arms if that stream stops.
const $ = (id) => document.getElementById(id);
const SEND_PERIOD_MS = 50;
const DEADZONE = 0.15;
// Standard Gamepad API button indices.
const BTN = { B: 1, Y: 3, LB: 4, RB: 5, RT: 7, START: 9 };

let ws, hello, state, prevUrl, seq = 0, controlling = false;
let arm = "left";
const keys = new Set();
let prevPressed = [];

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => setLink("connected", "active");
  ws.onclose = () => { setLink("disconnected", "estop"); controlling = false; updateButtons(); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onFrame(ev.data));
}

function send(msg) { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }
function setLink(text, cls) { $("link").textContent = text; $("link").className = `pill ${cls}`; }

function onFrame(blob) {
  const url = URL.createObjectURL(blob);
  $("video").src = url;
  if (prevUrl) URL.revokeObjectURL(prevUrl);
  prevUrl = url;
}

function onJson(m) {
  if (m.type === "hello") { hello = m; buildJoints(); }
  else if (m.type === "state") onState(m);
  else if (m.type === "pong") $("rtt").textContent = `RTT ${(performance.now() - m.t).toFixed(1)} ms`;
  else if (m.type === "control") { controlling = m.granted; if (!m.granted) alert("Another operator has control."); updateButtons(); }
  else if (m.type === "error") console.warn("edge error", m);
}

function onState(m) {
  state = m;
  controlling = m.controller === hello?.client_id;
  $("mode").textContent = m.mode.toUpperCase();
  $("mode").className = `pill ${m.mode}`;
  $("reason").textContent = m.reason;
  m.q.forEach((q, i) => { $(`meas${i}`).textContent = `${fmt(q)} → ${fmt(m.target[i])}`; });
  showTask(m.task);
  updateButtons();
}

function showTask(t) {
  const done = !t.active && t.results.length === t.total;
  $("t-index").textContent = t.active ? `${t.index + 1} / ${t.total}` : done ? `done, ${sum(t.results).toFixed(1)} s total` : "—";
  $("t-elapsed").textContent = t.active ? `${t.elapsed.toFixed(1)} s` : "—";
  $("t-dist").textContent = t.active && t.dist != null ? `${(t.dist * 100).toFixed(1)} cm` : "—";
  $("t-results").innerHTML = t.results.map((s, i) => `<span>${i + 1}: ${s.toFixed(1)} s</span>`).join("");
}

function buildJoints() {
  $("joints").innerHTML = hello.names.map((name, i) =>
    `<div class="joint"><span>${name.replace("_ctrl", "")}</span><code id="meas${i}">—</code></div>`).join("");
}

function updateButtons() {
  const active = state?.mode === "active";
  $("claim").textContent = controlling ? "Release control" : "Claim control";
  $("activate").disabled = !controlling || active;
  $("home").disabled = !controlling || !active;
  $("task").disabled = !controlling;
}

// Returns {v: [right, forward, up] in -1..1, grip: 0..1} from gamepad or keyboard.
function readInput() {
  const pad = [...(navigator.getGamepads?.() ?? [])].find(Boolean);
  let v = [0, 0, 0], grip = 0;
  if (pad) {
    const dz = (x) => (Math.abs(x) < DEADZONE ? 0 : x);
    v = [dz(pad.axes[0]), -dz(pad.axes[1]), -dz(pad.axes[3] ?? 0)];
    grip = pad.buttons[BTN.RT]?.value ?? 0;
    const pressed = pad.buttons.map((b) => b.pressed);
    const edge = (i) => pressed[i] && !prevPressed[i];
    if (edge(BTN.LB)) arm = "left";
    if (edge(BTN.RB)) arm = "right";
    if (edge(BTN.B)) send({ type: "estop" });
    if (edge(BTN.START)) send({ type: "activate" });
    if (edge(BTN.Y)) send({ type: "home" });
    prevPressed = pressed;
  }
  const k = (a, b) => (keys.has(a) ? 1 : 0) - (keys.has(b) ? 1 : 0);
  v = [v[0] || k("KeyD", "KeyA"), v[1] || k("KeyW", "KeyS"), v[2] || k("KeyR", "KeyF")];
  if (keys.has("KeyG")) grip = 1;
  $("device").textContent = pad ? pad.id.slice(0, 40) : "keyboard";
  return { v, grip };
}

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); send({ type: "estop" }); return; }
  if (e.repeat) return;
  if (e.code === "KeyQ") arm = "left";
  else if (e.code === "KeyE") arm = "right";
  else if (e.code === "Enter") send({ type: "activate" });
  else if (e.code === "KeyH") send({ type: "home" });
  keys.add(e.code);
});
document.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());

const fmt = (v) => v.toFixed(3).padStart(7);
const sum = (a) => a.reduce((x, y) => x + y, 0);

$("claim").onclick = () => send({ type: controlling ? "release" : "claim" });
$("activate").onclick = () => send({ type: "activate" });
$("home").onclick = () => send({ type: "home" });
$("estop").onclick = () => send({ type: "estop" });
$("task").onclick = () => send({ type: "task_start" });

setInterval(() => {
  const { v, grip } = readInput();
  $("arm").textContent = arm;
  $("stick").textContent = v.map((x) => x.toFixed(2)).join(" ");
  $("grip").style.width = `${grip * 100}%`;
  if (controlling && state?.mode === "active" && !$("pause").checked) {
    send({ type: "ee", seq: ++seq, arm, v, grip });
  }
}, SEND_PERIOD_MS);
setInterval(() => send({ type: "ping", t: performance.now() }), 1000);

connect();
