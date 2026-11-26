// Operator UI. While this client controls an active robot it streams the
// gamepad (or keyboard) state at 20 Hz as end-effector velocities; the edge
// holds the arms if that stream stops.
import { createView3D } from "./view3d.js";

const $ = (id) => document.getElementById(id);
const SEND_PERIOD_MS = 50;
const DEADZONE = 0.15;
// Standard Gamepad API button indices. Xbox names; PS4: B = ○, X = □, Y = △, LB/RB = L1/R1, RT = R2.
const BTN = { B: 1, X: 2, Y: 3, LB: 4, RB: 5, RT: 7, START: 9, UP: 12, DOWN: 13, LEFT: 14, RIGHT: 15 };

let ws, hello, state, prevUrl, seq = 0, controlling = false;
let view3d = null, view = "3d";
let arm = "left", mode = "arm"; // mode: "arm" moves a gripper, "drive" moves the base
const keys = new Set();
let prevPressed = [];

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => { setLink("connected", "active"); send({ type: "view", video: view === "cam" }); };
  ws.onclose = () => { setLink("disconnected", "estop"); controlling = false; updateButtons(); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onFrame(ev.data));
}

function send(msg) { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }
function setLink(text, cls) { $("link").textContent = text; $("link").className = `pill ${cls}`; }

function onFrame(blob) {
  const url = URL.createObjectURL(blob);
  $("video").src = url;
  if (prevUrl) URL.revokeObjectURL(prevUrl);
  prevUrl = url;
}

function onJson(m) {
  if (m.type === "hello") { hello = m; buildJoints(); init3D(); $("t-far").hidden = !hello.base; }
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
  if (m.base) $("base").textContent = `x ${m.base[0].toFixed(2)}  y ${m.base[1].toFixed(2)}  yaw ${(m.base[2] * 180 / Math.PI).toFixed(0)}°`;
  if (view3d) {
    view3d.setBodies(m.bodies);
    view3d.setBerries(m.task.active ? m.task.index : -1, m.task.results.length);
  }
  updateButtons();
}

async function init3D() {
  if (view3d) return;
  try {
    view3d = await createView3D($("view3d"), hello);
    if (state) { view3d.setBodies(state.bodies); view3d.setPreset("robot"); }
  } catch (e) {
    console.error("3D view unavailable", e);
    $("viewhint").textContent = "3D view failed to load (needs WebGL and the three.js CDN).";
    setView("cam");
  }
}

// The stick follows whichever view is shown: the fixed robot camera, or the
// 3D view's current heading (sent as yaw).
function setView(v) {
  view = v;
  try { localStorage.setItem("view", v); } catch {}
  document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("video").hidden = v !== "cam";
  $("view3d").hidden = v !== "3d";
  $("presets").hidden = v !== "3d";
  send({ type: "view", video: v === "cam" }); // the edge renders camera frames only for viewers
  if (!$("viewhint").textContent.startsWith("3D view failed")) {
    $("viewhint").textContent = v === "3d"
      ? "Drag to orbit, scroll to zoom, right-drag to pan. Stick directions follow this view."
      : "Stick directions follow this camera.";
  }
}

function showTask(t) {
  const done = !t.active && t.results.length === t.total;
  $("t-index").textContent = t.active ? `${t.index + 1} / ${t.total}` : done ? `done, ${sum(t.results).toFixed(1)} s total` : "—";
  $("t-elapsed").textContent = t.active ? `${t.elapsed.toFixed(1)} s` : "—";
  $("t-dist").textContent = t.active && t.dist != null ? `${(t.dist * 100).toFixed(1)} cm` : "—";
  $("t-status").textContent = t.active && t.status ? t.status : "—";
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

function setMode(m) { if (hello?.base || m === "arm") mode = m; }
function toggleMode() { setMode(mode === "arm" ? "drive" : "arm"); }

// Returns {v, grip, direct} from gamepad or keyboard. v is [right, forward, up]
// for an arm, [right, forward, turn right] for the base; each in -1..1.
// direct: base command from the arrow keys or D-pad, which drive the base in
// either mode (null when none is held).
function readInput() {
  const pad = [...(navigator.getGamepads?.() ?? [])].find(Boolean);
  let v = [0, 0, 0], grip = 0;
  if (pad) {
    const dz = (x) => (Math.abs(x) < DEADZONE ? 0 : x);
    v = [dz(pad.axes[0]), -dz(pad.axes[1]), mode === "drive" ? dz(pad.axes[2] ?? 0) : -dz(pad.axes[3] ?? 0)];
    grip = pad.buttons[BTN.RT]?.value ?? 0;
    const pressed = pad.buttons.map((b) => b.pressed);
    const edge = (i) => pressed[i] && !prevPressed[i];
    if (edge(BTN.LB)) { arm = "left"; setMode("arm"); }
    if (edge(BTN.RB)) { arm = "right"; setMode("arm"); }
    if (edge(BTN.B)) send({ type: "estop" });
    if (edge(BTN.START)) send({ type: "activate" });
    if (edge(BTN.X)) toggleMode();
    if (edge(BTN.Y)) send({ type: "home" });
    prevPressed = pressed;
  }
  const btn = (i) => (pad?.buttons[i]?.pressed ? 1 : 0);
  const fwd = btn(BTN.UP) - btn(BTN.DOWN) || k("ArrowUp", "ArrowDown");
  const turn = btn(BTN.RIGHT) - btn(BTN.LEFT) || k("ArrowRight", "ArrowLeft");
  const direct = hello?.base && (fwd || turn) ? [0, fwd, turn] : null;
  v = [v[0] || k("KeyD", "KeyA"), v[1] || k("KeyW", "KeyS"), v[2] || (mode === "drive" ? k("KeyE", "KeyQ") : k("KeyR", "KeyF"))];
  if (keys.has("KeyG")) grip = 1;
  $("device").textContent = pad ? pad.id.slice(0, 40) : "keyboard";
  return { v, grip, direct };
}
const k = (a, b) => (keys.has(a) ? 1 : 0) - (keys.has(b) ? 1 : 0);

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); send({ type: "estop" }); return; }
  if (e.code === "Tab") { e.preventDefault(); if (!e.repeat) toggleMode(); return; }
  if (e.code.startsWith("Arrow")) e.preventDefault(); // drive the base, don't scroll the page
  if (e.repeat) return;
  if (mode === "arm" && e.code === "KeyQ") arm = "left";
  else if (mode === "arm" && e.code === "KeyE") arm = "right";
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
$("rgbd-open").onclick = () => window.open("rgbd.html", "rgbd", "popup,width=840,height=480");
document.querySelectorAll("[data-arm]").forEach((b) => (b.onclick = () => { arm = b.dataset.arm; setMode("arm"); b.blur(); }));
document.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { setMode(b.dataset.mode); b.blur(); }));
document.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
document.querySelectorAll("[data-preset]").forEach((b) => (b.onclick = () => view3d?.setPreset(b.dataset.preset)));
try { view = localStorage.getItem("view") || "3d"; } catch {}
setView(view);

setInterval(() => {
  const { v, grip, direct } = readInput();
  document.querySelectorAll("[data-arm]").forEach((b) => b.classList.toggle("on", mode === "arm" && b.dataset.arm === arm));
  document.querySelectorAll("[data-mode]").forEach((b) => {
    b.classList.toggle("on", b.dataset.mode === mode);
    b.disabled = b.dataset.mode === "drive" && !hello?.base;
  });
  $("stick").textContent = v.map((x) => x.toFixed(2)).join(" ");
  $("grip").style.width = `${grip * 100}%`;
  if (controlling && state?.mode === "active" && !$("pause").checked) {
    const msg = direct ? { type: "base", seq: ++seq, v: direct }
      : mode === "drive" ? { type: "base", seq: ++seq, v } : { type: "ee", seq: ++seq, arm, v, grip };
    if (view === "3d" && view3d) msg.yaw = view3d.yaw();
    send(msg);
  }
}, SEND_PERIOD_MS);
setInterval(() => send({ type: "ping", t: performance.now() }), 1000);

connect();
