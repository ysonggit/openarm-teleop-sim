// Head RGB-D viewer: a view-only window on the /rgbd stream. Each binary
// message is: uint32 LE n | n bytes JPEG color | zlib(uint16 LE depth in mm).
// Depth is metric, so every pixel back-projects through the pinhole intrinsics
// into a colored point cloud.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const SHOW_MIN = 0.2, SHOW_MAX = 2.0; // m; depth color scale, display only

const LUT = turboLut();
let intr = null, depth = null, pixels = null, prevUrl = null, frames = 0, lastFpsT = performance.now();
let colorBy = "rgb";
try { colorBy = localStorage.getItem("rgbd-color") || "rgb"; } catch {}

// --- point cloud view -------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
$("cloud").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(getComputedStyle(document.body).getPropertyValue("--view-bg").trim() || "#dcdad2");
const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.01, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// The cloud lives in the sensor frame (three.js axes: x right, y up, z back),
// tilted by the mounting pitch so the floor shows level.
const rig = new THREE.Group();
scene.add(rig);
rig.add(new THREE.AxesHelper(0.1)); // the sensor: red x right, green y up, blue z back
const geom = new THREE.BufferGeometry();
const points = new THREE.Points(geom, new THREE.PointsMaterial({ size: 0.008, vertexColors: true }));
rig.add(points);

function sensorView() {
  camera.position.set(0, 0, 0.001);
  controls.target.set(0, 0, -1).applyMatrix4(rig.matrixWorld);
  camera.position.applyMatrix4(rig.matrixWorld);
  controls.update();
}
function sideView() {  // behind, right of and above the sensor (level axes), so depth shows at once
  controls.target.set(0, 0, -1).applyMatrix4(rig.matrixWorld);
  camera.position.set(0.55, 0.45, 0.6);
  controls.update();
}

new ResizeObserver(() => {
  const { clientWidth: w, clientHeight: h } = $("cloud");
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}).observe($("cloud"));
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// --- stream -----------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/rgbd`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setLink("connected", "active");
  ws.onclose = () => { setLink("disconnected", "estop"); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onHello(JSON.parse(ev.data)) : onFrame(ev.data));
}

function setLink(text, cls) { $("link").textContent = text; $("link").className = `pill ${cls}`; }

function onHello(m) {
  const first = !intr;
  intr = m;
  const n = m.width * m.height;
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setDrawRange(0, 0);
  rig.rotation.x = -(m.pitch ?? 0) * Math.PI / 180;
  rig.updateMatrixWorld();
  if (first) sideView();
  $("intr").textContent = `${m.width}×${m.height}  fx ${m.fx.toFixed(1)}  cx ${m.cx} cy ${m.cy}  valid ${m.near}–${m.far} m`;
  $("dmin").textContent = `${SHOW_MIN} m`;
  $("dmax").textContent = `${SHOW_MAX} m`;
  $("ramp").style.background = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map((t) => rgb(LUT[Math.round(t * 255)])).join(",")})`;
}

const decode = document.createElement("canvas");
const dctx = decode.getContext("2d", { willReadFrequently: true });

async function onFrame(buf) {
  if (!intr) return;
  const n = new DataView(buf).getUint32(0, true);
  const jpeg = new Blob([buf.slice(4, 4 + n)], { type: "image/jpeg" });
  const url = URL.createObjectURL(jpeg);
  $("color").src = url;
  if (prevUrl) URL.revokeObjectURL(prevUrl);
  prevUrl = url;

  const [bitmap, raw] = await Promise.all([
    createImageBitmap(jpeg),
    new Response(new Blob([buf.slice(4 + n)]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer(),
  ]);
  decode.width = intr.width;
  decode.height = intr.height;
  dctx.drawImage(bitmap, 0, 0);
  pixels = dctx.getImageData(0, 0, intr.width, intr.height).data;
  depth = new Uint16Array(raw);
  updateCloud();

  frames++;
  const now = performance.now();
  if (now - lastFpsT > 1000) { $("fps").textContent = `${(frames * 1000 / (now - lastFpsT)).toFixed(1)} fps`; frames = 0; lastFpsT = now; }
}

// Back-project each valid pixel: x = (u - cx) z / fx, y = (v - cy) z / fy (y down),
// then to three.js axes (y up, z back). Pixels with no return are skipped.
function updateCloud() {
  if (!depth || !pixels) return;
  const { width: W, fx, fy, cx, cy } = intr;
  const pos = geom.attributes.position.array, col = geom.attributes.color.array;
  let k = 0;
  for (let i = 0; i < depth.length; i++) {
    const mm = depth[i];
    if (!mm) continue;
    const z = mm / 1000, u = i % W, v = (i - u) / W;
    pos[k * 3] = (u - cx) * z / fx;
    pos[k * 3 + 1] = -(v - cy) * z / fy;
    pos[k * 3 + 2] = -z;
    const c = colorBy === "depth" ? LUT[Math.round(255 * clamp((z - SHOW_MIN) / (SHOW_MAX - SHOW_MIN)))]
      : [pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]];
    col[k * 3] = c[0] / 255; col[k * 3 + 1] = c[1] / 255; col[k * 3 + 2] = c[2] / 255;
    k++;
  }
  geom.setDrawRange(0, k);
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
  geom.computeBoundingSphere();
  $("npts").textContent = `(${k.toLocaleString()} points)`;
}

function setColorBy(c) {
  colorBy = c;
  try { localStorage.setItem("rgbd-color", c); } catch {}
  document.querySelectorAll("[data-color]").forEach((b) => b.classList.toggle("on", b.dataset.color === c));
  $("legend").hidden = c !== "depth";
  updateCloud();
}
document.querySelectorAll("[data-color]").forEach((b) => (b.onclick = () => setColorBy(b.dataset.color)));
$("sensor-view").onclick = sensorView;
setColorBy(colorBy);

// Hover the color image: depth and the 3D point (OpenCV camera frame).
$("color").onmousemove = (ev) => {
  if (!intr || !depth) return;
  const r = ev.target.getBoundingClientRect();
  const u = Math.floor((ev.clientX - r.left) / r.width * intr.width);
  const v = Math.floor((ev.clientY - r.top) / r.height * intr.height);
  const mm = depth[v * intr.width + u];
  if (!mm) { $("readout").textContent = `pixel (${u}, ${v})  no return`; return; }
  const z = mm / 1000, x = (u - intr.cx) * z / intr.fx, y = (v - intr.cy) * z / intr.fy;
  $("readout").textContent = `pixel (${u}, ${v})  depth ${mm} mm  point x ${x.toFixed(3)}  y ${y.toFixed(3)}  z ${z.toFixed(3)} m`;
};

function clamp(t) { return Math.min(1, Math.max(0, t)); }
function rgb(c) { return `rgb(${c[0]},${c[1]},${c[2]})`; }

// Turbo colormap (polynomial fit by Google AI), near = blue, far = red.
function turboLut() {
  const poly = (t, k) => k.reduceRight((acc, c) => acc * t + c, 0);
  const R = [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943];
  const G = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604];
  const B = [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973];
  return Array.from({ length: 256 }, (_, i) => {
    const t = i / 255;
    return [R, G, B].map((k) => Math.round(255 * clamp(poly(t, k))));
  });
}

connect();
