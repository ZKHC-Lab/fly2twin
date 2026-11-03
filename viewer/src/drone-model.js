import * as THREE from "three";
import { GLTFLoader } from "../vendor/three/examples/jsm/loaders/GLTFLoader.js";

// Built-in quadrotor: simple license-free geometry, used unless a GLB is configured.
function quadrotor() {
  const group = new THREE.Group();
  const carbon = new THREE.MeshStandardMaterial({ color: 0x24282c, roughness: 0.55, metalness: 0.35 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8bec4, roughness: 0.35, metalness: 0.8 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.03, 0.13), carbon);
  group.add(body);
  const arm = 0.134, motors = [];
  for (const [x, z] of [[arm, arm], [-arm, arm], [arm, -arm], [-arm, -arm]]) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.012, Math.hypot(x, z)), carbon);
    bar.position.set(x / 2, 0, z / 2);
    bar.rotation.y = Math.atan2(x, z);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.022, 20), metal);
    motor.position.set(x, 0.016, z);
    group.add(bar, motor);
    motors.push([x, 0.028, z]);
  }
  // Nose is -Z: white light front, red light rear.
  for (const [z, color] of [[-0.082, 0xffffff], [0.082, 0xff2020]]) {
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.3 }));
    led.position.set(0, 0, z);
    group.add(led);
  }
  return { scene: group, meta: { motor_centers: motors, rotor_radius_m: 0.095 } };
}

// Model-local correction stays separate from the flight controller pose/tilt.
export async function loadDroneModel(options = {}, manifestUrl = location.href) {
  const params = new URLSearchParams(location.search);
  const base = new URL(manifestUrl, location.href);
  const url = params.has('droneUrl') ? new URL(params.get('droneUrl'), location.href).href
    : options.url ? new URL(options.url, base).href : null;
  const scale = Number(params.get('droneScale') ?? options.scale ?? 1);
  const rotationDeg = params.has('droneRotation') ? params.get('droneRotation').split(',').map(Number)
    : options.rotationDeg ?? [0, 0, 0];
  const offset = params.has('droneOffset') ? params.get('droneOffset').split(',').map(Number)
    : options.offset ?? [0, 0, 0];
  if (!Number.isFinite(scale) || scale <= 0 ||
      ![rotationDeg, offset].every(a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite)))
    throw Error('Invalid drone scale, rotationDeg or offset');
  const metaUrl = options.metaUrl ? new URL(options.metaUrl, base).href : null;
  const [gltf, meta] = url ? await Promise.all([
    new GLTFLoader().loadAsync(url),
    metaUrl ? fetch(metaUrl).then(r => {
      if (!r.ok) throw Error(`Drone metadata: HTTP ${r.status}`);
      return r.json();
    }) : Promise.resolve({motor_centers: []}),
  ]) : (() => { const q = quadrotor(); return [{ scene: q.scene }, q.meta]; })();
  const group = new THREE.Group();
  group.add(gltf.scene);
  group.scale.setScalar(scale);
  group.rotation.set(...rotationDeg.map(THREE.MathUtils.degToRad), 'XYZ');
  group.position.fromArray(offset);
  group.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = false;
      o.receiveShadow = false;
      if (o.material) {
        o.material.envMapIntensity = 0.6;
        if (o.material.emissive?.getHex()) o.material.emissiveIntensity = 1.3;
      }
    }
  });
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d"),
    image = ctx.createImageData(128, 128);
  for (let y = 0; y < 128; y++)
    for (let x = 0; x < 128; x++) {
      const dx = (x - 63.5) / 64,
        dy = (y - 63.5) / 64,
        r = Math.hypot(dx, dy),
        a = Math.atan2(dy, dx),
        i = (y * 128 + x) * 4;
      const alpha =
        Math.max(0, 1 - r * r) *
        (0.14 + 0.1 * Math.pow(Math.cos(a * 2 + r * 5), 2));
      image.data.set([160, 174, 182, Math.round(255 * alpha)], i);
    }
  ctx.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const rotors = [];
  for (const p of meta.motor_centers) {
    const disk = new THREE.Mesh(
      new THREE.PlaneGeometry(meta.rotor_radius_m * 2, meta.rotor_radius_m * 2),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        opacity: 0.75,
      }),
    );
    disk.position.fromArray(p);
    disk.position.y += 0.001;
    disk.rotation.x = -Math.PI / 2;
    group.add(disk);
    rotors.push(disk);
  }
  let meshCount = 0;
  gltf.scene.traverse(o => { if (o.isMesh) meshCount++; });
  return { group, rotors, meta, status: {loaded: true, url: url ?? 'built-in quadrotor', scale, rotationDeg, offset, meshCount} };
}
