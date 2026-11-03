// Operator UI. Streams the slider goal at 20 Hz while this client controls an
// active robot; the edge holds the arm if that stream stops.
const $ = (id) => document.getElementById(id);
const CMD_PERIOD_MS = 50;

let ws, hello, goal, state, prevUrl, seq = 0, controlling = false;

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
  if (m.type === "hello") { hello = m; buildSliders(); }
  else if (m.type === "state") onState(m);
  else if (m.type === "pong") $("rtt").textContent = `RTT ${(performance.now() - m.t).toFixed(1)} ms`;
  else if (m.type === "control") { controlling = m.granted; if (!m.granted) alert("Another operator has control."); updateButtons(); }
  else if (m.type === "error") console.warn("edge error", m);
}

function onState(m) {
  const wasActive = state?.mode === "active";
  state = m;
  controlling = m.controller === hello?.client_id;
  // Entering ACTIVE: start the goal at the edge's setpoint so nothing jumps.
  if (m.mode === "active" && !wasActive) setGoal(m.target);
  $("mode").textContent = m.mode.toUpperCase();
  $("mode").className = `pill ${m.mode}`;
  $("reason").textContent = m.reason;
  m.q.forEach((q, i) => { $(`meas${i}`).textContent = `${fmt(q)} → ${fmt(m.target[i])}`; });
  updateButtons();
}

function buildSliders() {
  const box = $("joints");
  box.innerHTML = "";
  let arm;
  hello.names.forEach((name, i) => {
    const lo = hello.lower[i], hi = hello.upper[i];
    const [, side, joint] = name.replace("_ctrl", "").match(/^(?:(left|right)_)?(.*)$/);
    if (side && side !== arm) {
      arm = side;
      box.insertAdjacentHTML("beforeend", `<h3>${side[0].toUpperCase() + side.slice(1)} arm</h3>`);
    }
    box.insertAdjacentHTML("beforeend",
      `<div class="joint"><span>${joint}</span>
       <input type="range" id="s${i}" min="${lo}" max="${hi}" step="${(hi - lo) / 1000}">
       <code id="meas${i}">—</code></div>`);
    $(`s${i}`).addEventListener("input", (e) => { if (goal) goal[i] = +e.target.value; });
  });
  setGoal(hello.home);
}

function setGoal(q) {
  goal = [...q];
  goal.forEach((v, i) => { $(`s${i}`).value = v; });
}

function updateButtons() {
  const active = state?.mode === "active";
  $("claim").textContent = controlling ? "Release control" : "Claim control";
  $("activate").disabled = !controlling || active;
  $("home").disabled = !controlling || !active;
  hello?.names.forEach((_, i) => { $(`s${i}`).disabled = !controlling || !active; });
}

const fmt = (v) => v.toFixed(3).padStart(7);

$("claim").onclick = () => send({ type: controlling ? "release" : "claim" });
$("activate").onclick = () => send({ type: "activate" });
$("home").onclick = () => setGoal(hello.home);
$("estop").onclick = () => send({ type: "estop" });
document.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); send({ type: "estop" }); } });

setInterval(() => {
  if (controlling && state?.mode === "active" && goal && !$("pause").checked) {
    send({ type: "cmd", seq: ++seq, q: goal });
  }
}, CMD_PERIOD_MS);
setInterval(() => send({ type: "ping", t: performance.now() }), 1000);

connect();
