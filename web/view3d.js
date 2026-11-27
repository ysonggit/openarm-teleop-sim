// 3D view drawn in the browser from body poses in each state message (like
// RViz from /joint_states). Orbit, zoom and pan never wait on the edge.
// On a mobile base the view slides along with the base (no rotation, so the
// stick directions don't spin under the operator); presets are set relative
// to the base.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const deg = Math.PI / 180;

// sceneName: another scene's geometry (episode replay); default: the one the robot is in now.
export async function createView3D(container, hello, sceneName) {
  const q = sceneName ? `?scene=${encodeURIComponent(sceneName)}` : "";
  const [meta, blob] = await Promise.all([
    fetch(`scene.json${q}`).then((r) => r.json()),
    fetch(`scene.bin${q}`).then((r) => r.arrayBuffer()),
  ]);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getComputedStyle(document.body).getPropertyValue("--view-bg").trim() || "#dcdad2");
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8170, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(1, -1, 3);
  scene.add(sun);
  const floor = meta.geoms.find((g) => g.type === "plane");
  if (floor && meta.ground !== "soil") { // 10 cm cells over the floor; a soil scene shows the soil instead
    const size = 2 * Math.max(floor.size[0], floor.size[1]);
    const grid = new THREE.GridHelper(size, Math.round(size * 10), 0x6b6a64, 0x9b978a);
    grid.rotateX(Math.PI / 2);
    grid.position.set(floor.pos[0], floor.pos[1], 0.001);
    scene.add(grid);
  }
  const soil = soilTexture();

  const bodies = Array.from({ length: meta.nbody }, () => new THREE.Group());
  scene.add(...bodies);
  const facesStart = meta.nvert * 12;
  const meshGeom = meta.meshes.map(({ v, f }) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(blob, v[0] * 12, v[1] * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, facesStart + f[0] * 12, f[1] * 3), 1));
    g.computeVertexNormals();
    return g;
  });

  const byId = new Map();
  for (const gm of meta.geoms) {
    const isSoil = gm.name.startsWith("soil");
    const geometry = isSoil ? withGroundUV(makeGeometry(gm, meshGeom).clone(), gm) : makeGeometry(gm, meshGeom);
    const [r, g, b, a] = gm.rgba;
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(r, g, b), opacity: a, transparent: a < 1,
      roughness: isSoil ? 1 : 0.75, metalness: isSoil ? 0 : 0.1, side: THREE.DoubleSide,
      ...(isSoil ? { map: soil, bumpMap: soil, bumpScale: 2 } : {}),
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.fromArray(gm.pos);
    const [qw, qx, qy, qz] = gm.quat;
    mesh.quaternion.set(qx, qy, qz, qw);
    bodies[gm.body].add(mesh);
    byId.set(gm.id, mesh);
  }

  const camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.02, 20);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minPolarAngle = 0.02; // keep a heading so the stick always has a "forward"
  controls.maxPolarAngle = Math.PI / 2 - 0.02; // stay above the floor
  controls.maxDistance = 6;

  // Presets in the base frame (the base starts at the world origin).
  const robotCam = hello.camera;
  const z = robotCam.lookat[2] - 0.45; // mount height of the arms
  const presets = {
    robot: robotCam,
    front: { lookat: [0.3, 0, 0.45 + z], distance: 1.6 + z, azimuth: 180, elevation: -15 },
    left: { lookat: [0.3, 0, 0.45 + z], distance: 1.6 + z, azimuth: -90, elevation: -15 },
    right: { lookat: [0.3, 0, 0.45 + z], distance: 1.6 + z, azimuth: 90, elevation: -15 },
    top: { lookat: [0.35, 0, 0.3 + z], distance: 1.6 + z, azimuth: 0, elevation: -89 },
  };
  const baseBody = hello.base?.body;
  let base = { x: 0, y: 0, yaw: 0 };
  // MuJoCo free-camera convention: the camera looks along (azimuth, elevation).
  function setPreset(name) {
    const { lookat, distance, azimuth, elevation } = presets[name];
    const c = Math.cos(base.yaw), s = Math.sin(base.yaw);
    const L = [base.x + c * lookat[0] - s * lookat[1], base.y + s * lookat[0] + c * lookat[1], lookat[2]];
    const a = azimuth * deg + base.yaw, e = elevation * deg;
    controls.target.fromArray(L);
    camera.position.set(
      L[0] - distance * Math.cos(e) * Math.cos(a),
      L[1] - distance * Math.cos(e) * Math.sin(a),
      L[2] - distance * Math.sin(e),
    );
    controls.update();
  }
  setPreset("robot");

  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = container;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }).observe(container);

  renderer.setAnimationLoop(() => {
    if (container.hidden) return;
    controls.update();
    renderer.render(scene, camera);
  });

  const colors = hello.berries;
  return {
    setPreset,
    setBodies(flat) {
      bodies.forEach((b, i) => {
        const o = i * 7;
        b.position.set(flat[o], flat[o + 1], flat[o + 2]);
        b.quaternion.set(flat[o + 4], flat[o + 5], flat[o + 6], flat[o + 3]);
      });
      if (baseBody == null) return;
      const o = baseBody * 7;
      const next = { x: flat[o], y: flat[o + 1], yaw: 2 * Math.atan2(flat[o + 6], flat[o + 3]) };
      const d = new THREE.Vector3(next.x - base.x, next.y - base.y, 0);
      camera.position.add(d);
      controls.target.add(d);
      base = next;
    },
    setBerries(active, done) {
      colors.ids.forEach((id, i) => {
        const rgba = i === active ? colors.active : i < done ? colors.done : colors.idle;
        const m = byId.get(id)?.material;
        if (m) { m.color.setRGB(rgba[0], rgba[1], rgba[2]); m.opacity = rgba[3]; }
      });
    },
    // Heading of the view in the world XY plane (rad from +x), for the stick.
    yaw() {
      const d = controls.target.clone().sub(camera.position);
      return Math.atan2(d.y, d.x);
    },
  };
}

// Texture coordinates from world x, y (every 0.5 m), for the soil mesh, which has none.
function withGroundUV(geometry, gm) {
  const [qw, qx, qy, qz] = gm.quat;
  const m = new THREE.Matrix4().compose(new THREE.Vector3(...gm.pos), new THREE.Quaternion(qx, qy, qz, qw), new THREE.Vector3(1, 1, 1));
  const p = geometry.attributes.position, v = new THREE.Vector3(), uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).applyMatrix4(m);
    uv[2 * i] = v.x / 0.5; uv[2 * i + 1] = v.y / 0.5;
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  return geometry;
}

// Procedural soil: brown base, dark and light crumbs, a few pebbles. Tiles seamlessly.
function soilTexture() {
  const N = 256, c = document.createElement("canvas");
  c.width = c.height = N;
  const g = c.getContext("2d");
  g.fillStyle = "#5c4532";
  g.fillRect(0, 0, N, N);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const dot = (x, y, r, color) => {
    g.fillStyle = color;
    for (const dx of [-N, 0, N]) for (const dy of [-N, 0, N]) { g.beginPath(); g.arc(x + dx, y + dy, r, 0, 2 * Math.PI); g.fill(); }
  };
  for (let i = 0; i < 900; i++) dot(rnd() * N, rnd() * N, 1 + rnd() * 5, `rgba(${rnd() < 0.5 ? "40,28,18" : "120,95,70"},${0.15 + rnd() * 0.35})`);
  for (let i = 0; i < 2500; i++) dot(rnd() * N, rnd() * N, 0.6 + rnd(), `rgba(30,20,12,${0.2 + rnd() * 0.4})`);
  for (let i = 0; i < 25; i++) dot(rnd() * N, rnd() * N, 1.5 + rnd() * 2, `rgba(150,140,125,${0.5 + rnd() * 0.4})`);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function makeGeometry(gm, meshGeom) {
  const s = gm.size;
  switch (gm.type) {
    case "mesh": return meshGeom[gm.mesh];
    case "sphere": return new THREE.SphereGeometry(s[0], 24, 16);
    case "ellipsoid": return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "cylinder": return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
    case "capsule": return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
    case "plane": return new THREE.PlaneGeometry(2 * (s[0] || 2), 2 * (s[1] || 2));
  }
}
