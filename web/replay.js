// Episode replay: pick a recorded attempt, scrub through it in 3D, and see who acted when,
// what was commanded, and how it ended. Reads /episodes and /episodes/<id> from the edge.
import { createView3D } from "./view3d.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SCENE_NAMES = { strawberry: "Strawberries", weeding: "Weeding" };
const DIST_MAX = 0.5; // m, top of the distance chart

let all = [], ep = null, d = null, i = 0, playing = false, lastFrame = 0;
const views = new Map(); // scene -> {el, view}

async function loadList() {
  all = await fetch("episodes").then((r) => r.json());
  const scenes = [...new Set(all.map((e) => e.scene))];
  const sel = $("f-scene"), keep = sel.value;
  sel.innerHTML = `<option value="">All scenes</option>` + scenes.map((s) => `<option value="${s}">${SCENE_NAMES[s] ?? s}</option>`).join("");
  sel.value = scenes.includes(keep) ? keep : "";
  renderList();
}

function renderList() {
  const fs = $("f-scene").value, fr = $("f-result").value, fh = $("f-help").checked;
  const rows = all.filter((e) => (!fs || e.scene === fs) && (!fr || e.outcome.result === fr) && (!fh || e.actors.includes("human")));
  const done = rows.filter((e) => e.outcome.result !== "aborted");
  const ok = done.filter((e) => e.outcome.result === "success");
  const helped = done.filter((e) => e.actors.includes("human"));
  $("stats").textContent = rows.length ? `${rows.length} episodes · ${done.length ? Math.round(100 * ok.length / done.length) : 0}% success of finished · `
    + `${helped.length} with operator help · ${rows.reduce((n, e) => n + e.tickets.length, 0)} tickets` : "";
  $("empty").hidden = all.length > 0;
  $("episodes").tBodies[0].innerHTML = rows.map((e) => `<tr data-id="${e.id}" class="${ep?.id === e.id ? "on" : ""}">
      <td>${e.started_at.slice(5, 19).replace("T", " ")}</td><td>${SCENE_NAMES[e.scene] ?? e.scene}</td><td>${e.target}</td>
      <td><span class="badge ${e.outcome.result}">${e.outcome.result}</span>${e.tickets.length ? " ▼" : ""}</td>
      <td>${e.actors.map((a) => `<span class="who ${a}">${a}</span>`).join(" ")}</td><td>${e.duration_s.toFixed(1)}</td></tr>`).join("");
  document.querySelectorAll("#episodes tbody tr").forEach((tr) => (tr.onclick = () => open(tr.dataset.id)));
}

async function open(id) {
  ep = await fetch(`episodes/${encodeURIComponent(id)}`).then((r) => r.json());
  d = derive(ep);
  history.replaceState(null, "", `#${id}`);
  renderList();
  $("detail").hidden = false;
  $("download").href = `episodes/${encodeURIComponent(id)}`;
  $("download").setAttribute("download", `${id}.json`);
  const o = ep.outcome;
  $("meta").innerHTML = [
    ["Scene", SCENE_NAMES[ep.scene] ?? ep.scene], ["Task", ep.task], ["Target", `${ep.target.name} (#${ep.target.index + 1})`],
    ["Started", ep.started_at.replace("T", " ").replace("Z", " UTC")], ["Duration", `${ep.duration_s.toFixed(1)} s`],
    ["Result", `<span class="badge ${o.result}">${o.result}</span> by ${o.by ?? "—"}: ${o.reason}`],
    ["Steps", `${d.n} @ ${ep.rate_hz} Hz`],
  ].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
  $("events").innerHTML = d.marks.map((m) => `<li data-t="${m.t}" class="${m.ticket ? "ticket" : ""}">${m.t.toFixed(1)} s · ${m.text}</li>`).join("")
    || "<li>none</li>";
  document.querySelectorAll("#events li[data-t]").forEach((li) => (li.onclick = () => seek(stepAt(+li.dataset.t))));
  $("scrub").max = d.n - 1;
  await showScene(ep);
  seek(0);
  setPlaying(false);
}

// Per-step series for the charts, plus the marks (tickets and events) on the timeline.
function derive(ep) {
  const s = ep.steps, n = s.t.length;
  let grip = 0;
  const dist = [], grips = [], stick = [];
  for (let k = 0; k < n; k++) {
    const tg = s.target[k];
    dist.push(Math.min(...Object.values(s.tcp[k]).map((p) => Math.hypot(p[0] - tg[0], p[1] - tg[1], p[2] - tg[2]))));
    const a = s.action[k];
    if (a?.kind === "ee") grip = a.grip;
    grips.push(grip);
    stick.push(a?.v ? Math.min(1, Math.hypot(...a.v)) : 0);
  }
  const text = (e) => ({
    mode: `safety gate → ${e.mode} (${e.reason})`, detached: `${e.arm} gripper took it off the ${ep.scene === "weeding" ? "soil" : "stem"}`,
    takeover: `operator (client ${e.client}) took over: ${e.reason}`, release: `operator released: ${e.reason}`,
  }[e.kind] ?? `${e.kind}`);
  const marks = [...ep.tickets.map((k) => ({ t: k.t, ticket: true, text: `ticket "${k.code}": ${k.text}` })),
                 ...ep.events.map((e) => ({ t: e.t, text: text(e) }))].sort((a, b) => a.t - b.t);
  return { n, t: s.t, dist, grips, stick, marks, T: Math.max(ep.duration_s, s.t[n - 1] ?? 0, 0.1) };
}

async function showScene(ep) {
  for (const [, v] of views) v.el.hidden = true;
  let v = views.get(ep.scene);
  if (!v) {
    const el = document.createElement("div");
    el.className = "view3d";
    $("view3d").appendChild(el);
    v = { el, view: null };
    views.set(ep.scene, v);
    try {
      v.view = await createView3D(el, { ...ep.view, camera: ep.view.camera }, ep.scene);
      v.view.setPreset("robot");
    } catch (e) {
      el.textContent = "3D view unavailable (needs WebGL and the three.js CDN).";
    }
  }
  v.el.hidden = false;
  $("view3d").style.aspectRatio = "auto";
}

const stepAt = (t) => { let k = 0; while (k + 1 < d.n && d.t[k + 1] <= t) k++; return k; };

function seek(k) {
  i = Math.max(0, Math.min(d.n - 1, k));
  $("scrub").value = i;
  $("clock").textContent = `${(d.t[i] ?? 0).toFixed(1)} s`;
  const v = views.get(ep.scene)?.view;
  const frame = ep.bodies[Math.min(ep.bodies.length - 1, Math.floor(i * ep.bodies_hz / ep.rate_hz))];
  if (v && frame) {
    v.setBodies(frame);
    v.setBerries(ep.steps.attached[i] || ep.outcome.result !== "success" ? ep.target.index : -1, ep.target.index);
  }
  drawTimeline();
  drawChart();
  showNow();
}

function showNow() {
  const s = ep.steps, a = s.action[i], f = (p) => p.map((x) => x.toFixed(3)).join(" ");
  const act = !a ? "—" : a.kind === "ee" ? `${a.arm} arm · stick ${a.v.map((x) => x.toFixed(2)).join(" ")} · grip ${a.grip}`
    : a.kind === "base" ? `base · stick ${a.v.map((x) => x.toFixed(2)).join(" ")}` : a.kind;
  $("now").innerHTML = [
    ["t", `${s.t[i].toFixed(2)} s`], ["Actor", `<span class="who ${s.actor[i]}">${s.actor[i]}</span>`], ["Gate", s.mode[i]],
    ["Command", act], ["Base", `x ${s.base[i][0].toFixed(2)} y ${s.base[i][1].toFixed(2)} yaw ${(s.base[i][2] * 180 / Math.PI).toFixed(0)}°`],
    ["Gripper L", f(s.tcp[i].left)], ["Gripper R", f(s.tcp[i].right)], ["Target", f(s.target[i])],
    ["To target", `${(d.dist[i] * 100).toFixed(1)} cm`], ["Attached", s.attached[i] ? "yes" : "no, off"],
  ].map(([k, v]) => `<span>${k}</span><code>${v}</code>`).join("");
}

function fit(canvas) {
  const r = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.height / (canvas._r || 1);
  canvas._r = r;
  canvas.width = w * r;
  canvas.height = h * r;
  const g = canvas.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  return [g, w, h];
}
const X = (t, w) => (t / d.T) * w;

function drawTimeline() {
  const [g, w, h] = fit($("timeline"));
  for (const sgm of ep.segments) {
    g.fillStyle = css(`--${sgm.actor}`) || "#999";
    g.fillRect(X(sgm.t0, w), 14, Math.max(1, X(sgm.t1 - sgm.t0, w)), h - 18);
  }
  g.fillStyle = css("--fg");
  for (const m of d.marks) {
    const x = X(m.t, w);
    if (m.ticket) { g.fillStyle = css("--stop"); g.beginPath(); g.moveTo(x - 6, 0); g.lineTo(x + 6, 0); g.lineTo(x, 11); g.fill(); }
    else { g.fillStyle = css("--fg"); g.fillRect(x - 0.5, 14, 1, h - 18); }
  }
  g.fillStyle = css("--fg");
  g.fillRect(X(d.t[i], w) - 1, 0, 2, h);
}

function drawChart() {
  const [g, w, h] = fit($("chart"));
  const pad = 4, H = h - 2 * pad;
  g.strokeStyle = css("--line"); g.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75]) { g.beginPath(); g.moveTo(0, pad + H * f); g.lineTo(w, pad + H * f); g.stroke(); }
  const line = (ys, color, max) => {
    g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
    ys.forEach((y, k) => { const x = X(d.t[k], w), yy = pad + H * (1 - Math.min(y, max) / max); k ? g.lineTo(x, yy) : g.moveTo(x, yy); });
    g.stroke();
  };
  line(d.stick, css("--stick"), 1);
  line(d.grips, css("--grip"), 1);
  line(d.dist, css("--dist"), DIST_MAX);
  g.fillStyle = css("--muted"); g.font = "10px system-ui";
  g.fillText(`${DIST_MAX * 100} cm`, 4, 12);
  g.fillStyle = css("--fg");
  g.fillRect(X(d.t[i], w) - 1, 0, 2, h);
}

function setPlaying(p) {
  playing = p && d?.n > 1;
  $("play").textContent = playing ? "❚❚" : "▶";
  if (playing) { if (i >= d.n - 1) seek(0); lastFrame = performance.now(); requestAnimationFrame(loop); }
}
function loop(now) {
  if (!playing) return;
  const t = d.t[i] + ((now - lastFrame) / 1000) * +$("speed").value;
  lastFrame = now;
  let k = i;
  while (k + 1 < d.n && d.t[k + 1] <= t) k++;
  if (k !== i) seek(k);
  if (k >= d.n - 1) return setPlaying(false);
  requestAnimationFrame(loop);
}

function clickSeek(ev) {
  const r = ev.currentTarget.getBoundingClientRect();
  seek(stepAt(((ev.clientX - r.left) / r.width) * d.T));
}

$("play").onclick = () => setPlaying(!playing);
$("scrub").oninput = (e) => { setPlaying(false); seek(+e.target.value); };
$("timeline").onclick = clickSeek;
$("chart").onclick = clickSeek;
$("refresh").onclick = loadList;
["f-scene", "f-result", "f-help"].forEach((id) => ($(id).onchange = renderList));
document.addEventListener("keydown", (e) => {
  if (!d || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
  if (e.code === "ArrowRight") seek(i + (e.shiftKey ? 20 : 1));
  if (e.code === "ArrowLeft") seek(i - (e.shiftKey ? 20 : 1));
});
new ResizeObserver(() => { if (d) { drawTimeline(); drawChart(); } }).observe($("timeline"));

await loadList();
const wanted = decodeURIComponent(location.hash.slice(1));
if (wanted && all.some((e) => e.id === wanted)) open(wanted);
else if (all.length) open(all[0].id);
