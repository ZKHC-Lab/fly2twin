import * as THREE from "three";
import { loadSceneContract } from './scene-contract.js';
import { SplatMesh, SparkRenderer, SplatLoader } from "@sparkjsdev/spark";
import { PointPass } from "./point-pass.js";
import { CaptureRail } from "./capture-rail.js";
import { loadDroneModel } from "./drone-model.js";
import { Navigation } from "./navigation.js";
const $ = (id) => document.getElementById(id),
  V = (a) => new THREE.Vector3(...a),
  keys = new Set();
const params = new URLSearchParams(location.search),
  capture = params.has("capture");
if (params.has('noBackdrop')) {
  const css = document.createElement('style'); css.textContent = '* {backdrop-filter:none !important}'; document.head.append(css);
}
const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: "high-performance",
  alpha: true,
  preserveDrawingBuffer: capture,
});
const gl = renderer.getContext();
const gpuInfo = gl.getExtension("WEBGL_debug_renderer_info");
const gpuName = gpuInfo ? gl.getParameter(gpuInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
const softwareGPU = /swiftshader|llvmpipe|software|subzero/i.test(gpuName);
const discreteGPU = /nvidia|radeon rx|radeon pro|apple m[234]/i.test(gpuName);
const requestedQuality = params.get("quality");
let quality = ["low", "medium", "high"].includes(requestedQuality) ? requestedQuality
  : requestedQuality === "sh3" ? "high" : requestedQuality === "compact" ? "low"
  : discreteGPU ? "high" : "medium";
let resolutionScale = quality === "low" ? .75 : 1;
if (params.has("benchmark") && params.has("resolution")) resolutionScale = Math.max(.4, Math.min(1.5, Number(params.get("resolution")) || 1));
renderer.setPixelRatio(devicePixelRatio);
let drawingScale = 1, renderLoopRunning = true, idleSettled = false, lastMotionAt = performance.now();
let lastScaleChange = 0, currentFrameMs = 16.7, idlePaintTimer = null;
let slowScaleFrames = 0, fastScaleMs = 0, lastResizeAt = -Infinity, resizeCount = 0, resizeWallMs = 0;
let minResizeIntervalMs = Infinity, resizeTimer = null, viewportWidth = innerWidth, viewportHeight = innerHeight;
function noteResize(start) {
  if (Number.isFinite(lastResizeAt)) minResizeIntervalMs = Math.min(minResizeIntervalMs, start - lastResizeAt);
  lastResizeAt = start; resizeCount++; resizeWallMs += performance.now() - start;
}
const paintedPosition = new THREE.Vector3(Infinity, Infinity, Infinity), paintedQuaternion = new THREE.Quaternion();
function setDrawingScale(scale) {
  scale = Math.round(scale * 20) / 20;
  const ratio = devicePixelRatio * scale;
  if (scale === drawingScale && Math.abs(renderer.getPixelRatio() - ratio) < 1e-6) return true;
  const now = performance.now();
  if (now - lastResizeAt < 500) return false;
  drawingScale = scale;
  // three.js setPixelRatio calls setSize(width,height,false): no CSS writes.
  renderer.setPixelRatio(ratio); noteResize(now);
  return true;
}
function wakeRender() {
  // Input can wake an idle frame with dt almost zero. Give it time to move
  // before the idle detector is allowed to stop the animation loop again.
  lastMotionAt = performance.now();
  clearTimeout(idlePaintTimer);
  if (!renderLoopRunning) { idleSettled = false; renderLoopRunning = true; last = performance.now(); renderer.setAnimationLoop(animateFrame); }
}
window.viewerDevice = {renderer: gpuName, software: softwareGPU, quality};
const badge = document.createElement("span"); badge.className = "hud gpu-badge";
badge.textContent = /nvidia/i.test(gpuName) ? "NVIDIA" : /intel/i.test(gpuName) ? "Intel" : softwareGPU ? "软件渲染" : "GPU";
badge.title = gpuName; badge.style.cssText = "position:fixed;right:24px;bottom:18px;font-size:10px;color:#9eafb5;z-index:5";
document.body.append(badge);
renderer.setSize(innerWidth, innerHeight, false);
renderer.setClearColor(0, 0);
$("viewport").append(renderer.domElement);
const pointPass = new PointPass(renderer);
function renderScene() {
  if (ready) {
    const now = performance.now();
    const moving = flight || cruise || tween || openingRail !== null || camera.position.distanceToSquared(paintedPosition) > 1e-10 || 1 - Math.abs(camera.quaternion.dot(paintedQuaternion)) > 1e-10;
    if (moving) {
      wakeRender(); idleSettled = false; lastMotionAt = now;
      if (!params.has('fixedResolution')) {
        slowScaleFrames = currentFrameMs > 18 ? slowScaleFrames + 1 : 0;
        fastScaleMs = currentFrameMs < 12 ? fastScaleMs + currentFrameMs : 0;
        if (now - lastScaleChange >= 500 && now - lastResizeAt >= 500) {
          const ceiling = quality === "low" ? .75 : 1;
          if (slowScaleFrames >= 3) {
            resolutionScale = Math.max(discreteGPU ? .55 : .4, Math.round((resolutionScale - .05) * 20) / 20);
            slowScaleFrames = 0; fastScaleMs = 0; lastScaleChange = now;
          } else if (fastScaleMs >= 2500) {
            resolutionScale = Math.min(ceiling, Math.round((resolutionScale + .05) * 20) / 20);
            fastScaleMs = 0; lastScaleChange = now;
          }
        }
      }
      setDrawingScale(resolutionScale);
    } else if (now - lastMotionAt > 150 && renderLoopRunning && setDrawingScale(1)) {
      renderLoopRunning = false;
      // WebGLAnimation schedules its NEXT rAF after our callback returns.
      // Stopping synchronously here misses that request and duplicates loops on wake.
      queueMicrotask(() => { if (!renderLoopRunning) renderer.setAnimationLoop(null); });
      // A final settle draw lets Spark complete any already-submitted sort;
      // no repeated sorting or rasterization occurs during an idle inspection.
      const settle = () => {
        if (renderLoopRunning) return;
        // Spark 0.1.10 queues asynchronous accumulator uploads and WASM sorts.
        // Wait for both queues, not a fixed frame count, before preserving pixels.
        if (activeType === 'splat' && (spark.defaultView.sorting || spark.defaultView.pending || spark.pendingUpdate?.scene)) {
          idlePaintTimer = setTimeout(settle, 20); return;
        }
        if (activeType === 'points') pointPass.render(scene, camera);
        else renderer.render(scene, camera);
        idleSettled = true;
      };
      idlePaintTimer = setTimeout(settle, 80);
    }
    paintedPosition.copy(camera.position); paintedQuaternion.copy(camera.quaternion);
  }
  rimLight.visible = flight;
  rimLight.intensity = config?.droneRimIntensity ?? 1.4;
  if (flight) {
    rimLight.target.position.copy(drone.position);
    rimLight.position
      .copy(drone.position)
      .add(new THREE.Vector3(-1, 0.8, -0.8).applyQuaternion(camera.quaternion));
  }
  if (activeType === "points") pointPass.render(scene, camera);
  else renderer.render(scene, camera);
}
const scene = new THREE.Scene(),
  camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.03, 200);
const spark = new SparkRenderer({ renderer,
  maxStdDev: Math.sqrt(8),
  minAlpha: .5 / 255,
  view: {sortRadial: false, sortDistance: params.has('benchmark') && params.has('sortDistance') ? Number(params.get('sortDistance')) : .025, sort360: false},
});
scene.add(spark);
scene.add(new THREE.HemisphereLight(0xc5edff, 0x364255, 3));
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(3, 6, 1);
scene.add(light);
const rimLight = new THREE.DirectionalLight(0xcbd9e5, 1.4);
scene.add(rimLight, rimLight.target);
let navigation, rail,
  overviewTimer = null,
  openingRail = null,
  roll = 0;
let config,
  configs,
  activeAssetUrl = null,
  activeType = "splat",
  index = 0,
  root,
  cloud,
  line,
  path,
  lookPath,
  coverage = [],
  flight = false,
  guard = false,
  cruise = false,
  cruiseU = 0,
  tween = null,
  ready = false,
  yaw = 0,
  pitch = 0,
  heightColors = false;
let position = new THREE.Vector3(),
  velocity = new THREE.Vector3(),
  time = 0,
  last = performance.now(),
  generation = 0;
const drone = new THREE.Group(),
  tilt = new THREE.Group();
drone.add(tilt);
scene.add(drone);
drone.scale.setScalar(1);
drone.visible = false;
const rotors = [];
let droneModelStatus = {loaded: false};
const contact = new THREE.Mesh(
  new THREE.PlaneGeometry(0.7, 0.7),
  new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader:
      "varying vec2 uv0;void main(){uv0=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}",
    fragmentShader:
      "varying vec2 uv0;void main(){float r=length(uv0-.5);float ring=exp(-pow((r-.30)/.045,2.));float shadow=exp(-r*r*28.);gl_FragColor=vec4(mix(vec3(.01,.03,.04),vec3(.32,.85,.76),ring),(.16*shadow+.19*ring)*(1.-smoothstep(.43,.5,r)));}",
  }),
);
contact.rotation.x = -Math.PI / 2;
contact.visible = false;
scene.add(contact);
// A one-pixel dashed tether makes the ground projection readable without a beam.
const contactLinkGeometry = new THREE.BufferGeometry();
contactLinkGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
contactLinkGeometry.setAttribute('along', new THREE.BufferAttribute(new Float32Array([1, 0]), 1));
const contactLink = new THREE.Line(contactLinkGeometry, new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, depthTest: true,
  vertexShader: 'attribute float along; varying float vAlong; void main(){vAlong=along; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
  fragmentShader: 'varying float vAlong; void main(){if(fract(vAlong*24.)>.58)discard; gl_FragColor=vec4(.32,.85,.76,mix(.035,.25,vAlong));}',
}));
contactLink.visible = false;
scene.add(contactLink);
function transform(a) {
  let p = V(a);
  if (config.rotation)
    p.applyQuaternion(new THREE.Quaternion().fromArray(config.rotation));
  else if (config.up === "Z") p.set(p.x, p.z, -p.y);
  return p
    .multiplyScalar(config.scale ?? 1)
    .add(V(config.translation ?? [0, 0, 0]));
}
function quatAt(p, t) {
  const c = new THREE.PerspectiveCamera();
  c.position.copy(p);
  c.lookAt(t);
  return c.quaternion.clone();
}
function angles(q) {
  const e = new THREE.Euler().setFromQuaternion(q, "YXZ");
  yaw = e.y;
  pitch = e.x;
  roll = e.z;
}
function strictGS() {
  return config?.gsStrict && activeType === "splat";
}
function projectCamera() {
  if (activeType === 'splat' && config?.captureCalibration) {
    const {K, resolution:[w,h]} = config.captureCalibration;
    const n=camera.near, f=camera.far;
    camera.projectionMatrix.makePerspective(
      (-.5-K[0][2])/K[0][0]*n, (w-.5-K[0][2])/K[0][0]*n,
      (K[1][2]+.5)/K[1][1]*n, -(h-.5-K[1][2])/K[1][1]*n,n,f);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    const fit=Math.min(innerWidth/w,innerHeight/h), vw=Math.round(w*fit),vh=Math.round(h*fit);
    renderer.setSize(vw,vh,false);
    Object.assign(renderer.domElement.style,{position:'absolute',width:`${vw}px`,height:`${vh}px`,
      left:`${(innerWidth-vw)/2}px`,top:`${(innerHeight-vh)/2}px`});
    return;
  }
  camera.fov = activeType === "points" ? 58 : (config?.fov ?? 58);
  camera.updateProjectionMatrix();
  if (activeType === "splat" && config?.principalPoint) {
    camera.projectionMatrix.elements[8] = 1 - 2 * config.principalPoint[0];
    camera.projectionMatrix.elements[9] = 2 * config.principalPoint[1] - 1;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}
function constrain(p) {
  if (!guard) return navigation ? navigation.clamp(p) : p;
  if (config.bounds) {
    p.clamp(V(config.bounds.min), V(config.bounds.max));
  }
  if (coverage.length) {
    let best = coverage[0],
      d = Infinity;
    for (const c of coverage) {
      const x = p.distanceToSquared(c);
      if (x < d) {
        best = c;
        d = x;
      }
    }
    const r = activeType === "points" ? 0.5 : (config.corridorRadius ?? 0.5);
    if (d > r * r) p.sub(best).setLength(r).add(best);
  }
  return p;
}
function go(view, duration = 1.8, opening = false) {
  if (flight) position.copy(camera.position);
  flight = false;
  drone.visible = false;
  cruise = false;
  updateHUD();
  velocity.set(0, 0, 0);
  const end = constrain(transform(view.position)),
    q = view.quaternion
      ? rail.view(view).q
      : quatAt(end, transform(view.target));
  if (opening && strictGS()) {
    position.copy(end);
    camera.position.copy(end);
    camera.quaternion.copy(q);
    angles(q);
    if (!guard) {
      // Keep source world_up vertical in free roam instead of preserving the
      // capture camera's handheld roll. Guided/reference views stay calibrated.
      roll = 0;
      camera.rotation.set(pitch, yaw, roll, 'YXZ');
    }
    // Free roam starts at the contract's selected initial view, not the first
    // 1.8 seconds of the capture rail (which would silently replace that pose).
    openingRail = guard ? 0 : null;
    tween = null;
    return;
  }
  if (opening) {
    position.copy(end).add(new THREE.Vector3(0, 0.38, 0.12));
    constrain(position);
    camera.position.copy(position);
    camera.quaternion.copy(q);
  }
  tween = {
    start: position.clone(),
    end,
    q0: camera.quaternion.clone(),
    q,
    t: 0,
    duration,
  };
}
function clearRoot() {
  if (!root) return;
  root.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) {
      for (const m of Array.isArray(o.material) ? o.material : [o.material])
        m.dispose();
    }
  });
  cloud?.dispose?.();
  scene.remove(root);
}
async function load(
  i,
  requestedType = null,
  preserveView = false,
  cinematic = false,
) {
  const overview =
    false; // Never preload the 2.6M-point overview alongside Gaussian assets.
  overviewTimer = null;
  openingRail = null;
  const saved = preserveView
    ? {
        position: camera.position.clone(),
        quaternion: camera.quaternion.clone(),
      }
    : null;
  document.querySelectorAll("nav button").forEach((b) => (b.disabled = true));
  const g = ++generation;
  $("load-progress").value = 0;
  $("load-status").textContent = "下载场景";
  ready = false;
  window.viewerReady = false;
  if (!cinematic) {
    $("loading").classList.remove("done");
    renderer.domElement.style.opacity = 1;
  }
  $("error").hidden = true;
  index = i;
  config = configs[i];
  activeType = overview ? "points" : (requestedType ?? config.type);
  projectCamera();
  heightColors = false;
  contact.visible = false;
  flight = false;
  drone.visible = false;
  cruise = false;
  tween = null;
  guard = params.get("guided") !== "0";
  keys.clear();
  clearRoot();
  cloud = null;
  root = new THREE.Group();
  scene.add(root);
  $("title").textContent = config.name;
  $("subtitle").textContent = config.subtitle ?? "真实空间 · 自由探索";
  document
    .querySelectorAll("nav button")
    .forEach((b, j) => b.classList.toggle("active", b.dataset.group === (config.group ?? config.id)));
  try {
    const data = config.trajectoryData ?? (config.trajectory
      ? await fetch(config.trajectory).then((r) => {
          if (!r.ok) throw Error("trajectory");
          return r.json();
        })
      : { path: [config.initial], coverage: [] });
    if (g !== generation) return;
    navigation = new Navigation(config.navigationData ?? await fetch(`assets/${config.id}-navigation.json`).then(r => {
      if (!r.ok) throw Error("navigation asset unavailable"); return r.json();
    }));
    coverage = (data.coverage ?? []).map(transform);
    rail = new CaptureRail(data, config, transform);
    const samples = data.path.map((x) => transform(x.position));
    path =
      samples.length > 1
        ? new THREE.CatmullRomCurve3(samples, false, "centripetal")
        : null;
    lookPath = path
      ? new THREE.CatmullRomCurve3(
          data.path.map((x) => transform(x.target)),
          false,
          "centripetal",
        )
      : null;
    line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(samples),
      new THREE.LineBasicMaterial({
        color: 0x81ffe2,
        transparent: true,
        opacity: 0.55,
      }),
    );
    line.visible = false;
    root.add(line);
    if (activeType === "splat") {
      const scan = params.has("benchmark") && /^(350|500|650)$/.test(params.get("scan")) ? params.get("scan") : null;
      const assetUrl = scan ? `assets/${config.id}-scan${scan}.spz` : quality === "high" ? (config.highQualityUrl ?? config.url)
        : quality === "medium" ? config.mediumUrl : config.lowUrl;
      activeAssetUrl = assetUrl;
      const packed = await new SplatLoader().loadAsync(assetUrl, event => {
        if (g !== generation) return;
        if (event.total) $("load-progress").value = event.loaded / event.total * 90;
        $("load-status").textContent = event.loaded === event.total ? "解码与上传 GPU" : `下载 ${Math.round(event.loaded / 1048576)} MB`;
      });
      if (g !== generation) { packed.dispose(); return; }
      cloud = new SplatMesh({packedSplats: packed});
      cloud.maxSh = quality === "high" ? 3 : 2;
      root.add(cloud);
      if (config.rotation) cloud.quaternion.fromArray(config.rotation);
      else if (config.up === "Z") cloud.rotation.x = -Math.PI / 2;
      cloud.scale.setScalar(config.scale ?? 1);
      cloud.position.copy(V(config.translation ?? [0, 0, 0]));
      await cloud.initialized;
    } else {
      activeAssetUrl = config.pointUrl ?? config.url;
      const decoded = await new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./point-worker.js', import.meta.url), {type: 'module'});
        worker.onmessage = ({data}) => { worker.terminate(); data.error ? reject(Error(data.error)) : resolve(data); };
        worker.onerror = error => { worker.terminate(); reject(error); };
        worker.postMessage({...config, url: new URL(config.pointUrl ?? config.url, location.href).href});
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(decoded.positions, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(decoded.colors, 3));
      geo.computeBoundingBox();
      const colors = geo.attributes.color;
      geo.userData.originalColors = colors?.array.slice();
      const range = geo.boundingBox.max.y - geo.boundingBox.min.y;
      const mat = new THREE.ShaderMaterial({
        vertexColors: true,
        uniforms: {
          size: { value: config.pointSize ?? 0.018 },
          heightMode: { value: 0 },
          ymin: { value: geo.boundingBox.min.y },
          yrange: { value: range },
        },
        transparent: true,
        depthWrite: true,
        vertexShader: `
          uniform float size,heightMode,ymin,yrange;
          varying vec3 c;
          void main(){
            float h=clamp((position.y-ymin)/max(yrange,.001),0.,1.);
            vec3 hc=mix(vec3(.12,.65,.8),vec3(.95,.66,.32),h);
            c=mix(color,hc,heightMode);
            vec4 p=modelViewMatrix*vec4(position,1.);
            gl_Position=projectionMatrix*p;
            // Near samples stay fine; distant samples fill their projected gaps.
            gl_PointSize=mix(3.2,6.4,smoothstep(.8,8.,-p.z))*(size/.012);
          }`,
        fragmentShader: `
          varying vec3 c;
          void main(){
            float r=length(gl_PointCoord-.5);
            float alpha=1.-smoothstep(.30,.5,r);
            if(alpha<.08)discard;
            gl_FragColor=vec4(c,alpha);
            #include <tonemapping_fragment>
            #include <colorspace_fragment>
          }`,
      });
      cloud = new THREE.Points(geo, mat);
      root.add(cloud);
    }
    if (g !== generation) return;
    if (saved) {
      position.copy(saved.position);
      camera.position.copy(saved.position);
      camera.quaternion.copy(saved.quaternion);
      angles(saved.quaternion);
      velocity.set(0, 0, 0);
    } else if (overview) {
      guard = false;
      position.copy(transform(config.pointInitial.position));
      camera.position.copy(position);
      camera.quaternion.copy(
        quatAt(position, transform(config.pointInitial.target)),
      );
      angles(camera.quaternion);
      overviewTimer = 0;
    } else go(config.initial, 1.8, true);
    if (strictGS() && guard) {
      rail.constrain(position, camera.quaternion);
      camera.position.copy(position);
      angles(camera.quaternion);
    }
    if (cinematic)
      requestAnimationFrame(() => {
        renderer.domElement.style.opacity = 1;
      });
    if (activeType === "splat") {
      // Let Spark's own asynchronous sort finish; calling prepare concurrently
      // with its animation-loop update causes "Only one sort at a time".
      renderScene();
      while (!spark.defaultView.display) await new Promise(requestAnimationFrame);
    }
    renderScene();
    gl.finish(); // Include the first submitted frame in loading time.
    ready = true;
    adaptiveCooldown = performance.now() + (discreteGPU ? 12000 : 2000);
    window.viewerReady = true;
    $("load-progress").value = 100;
    $("quality").value = quality;
    const restored = sessionStorage.getItem("viewer-pose");
    sessionStorage.removeItem("viewer-pose");
    if (restored) {
      const pose = JSON.parse(restored);
      if (pose.id === config.id) {
        openingRail = null; tween = null;
        position.fromArray(pose.position); camera.position.copy(position);
        camera.quaternion.fromArray(pose.quaternion); angles(camera.quaternion);
      }
    }
    $("loading").classList.add("done");
    updateHUD();
  } catch (e) {
    console.error(e);
    $("loading").classList.add("done");
    $("error").hidden = false;
  } finally {
    document
      .querySelectorAll("nav button")
      .forEach((b) => (b.disabled = false));
  }
}
function updateHUD() {
  $("mode").textContent = cruise
    ? "自动巡航"
    : flight
      ? "惯性飞行"
      : "自由漫游";
  $("boundary").textContent =
    (guard ? "引导模式" : "自由漫游 · 包围盒限位") +
    (config.pointUrl ? (activeType === "splat" ? " · 3DGS" : " · 点云") : "");
  $("controls").innerHTML = flight
    ? "<b>W A S D</b> 水平推力 <b>Q E</b> 偏航 <b>Space / ⇧</b> 升降"
    : "<b>W A S D</b> 移动 <b>Q E</b> 转向 <b>R F</b> 升降 <b>⇧</b> 加速";
}
function toggleFlight() {
  flight = !flight;
  drone.visible = flight;
  velocity.set(0, 0, 0);
  tween = null;
  cruise = false;
  angles(camera.quaternion);
  if (!strictGS() && guard) pitch = 0;
  updateHUD();
}
function setCruise(u) {
  if (config.exactCruise && activeType === "splat") {
    const pose = rail.sample(u * config.cruiseDuration);
    position.copy(pose.p);
    camera.quaternion.copy(pose.q);
    angles(pose.q);
    return;
  }
  if (!path) return;
  const p = constrain(path.getPointAt(Math.min(u, 0.999999))),
    ahead = constrain(path.getPointAt(Math.min(u + 0.025, 1)));
  position.copy(p);
  const target = config.cruiseTarget
    ? transform(config.cruiseTarget)
    : lookPath
      ? lookPath.getPoint(path.getUtoTmapping(Math.min(u, 0.999999)))
      : ahead;
  if (target.distanceTo(p) < 0.02)
    target.copy(p).add(camera.getWorldDirection(new THREE.Vector3()));
  const q = quatAt(p, target);
  camera.quaternion.slerp(q, 0.045);
  angles(camera.quaternion);
}
function step(dt) {
  if (!ready) return;
  time += dt;
  if (overviewTimer !== null) {
    overviewTimer += dt;
    if (overviewTimer > 2) {
      overviewTimer = null;
      renderer.domElement.style.transition = "opacity .65s ease";
      renderer.domElement.style.opacity = 0;
      ready = false;
      window.viewerReady = false;
      const requestedGeneration = generation;
      setTimeout(() => {
        if (generation === requestedGeneration)
          load(index, "splat", false, true);
      }, 650);
    }
    renderScene();
    return;
  }
  if (openingRail !== null) {
    openingRail += dt;
    const pose = rail.sample(Math.min(openingRail, 1.8));
    position.copy(pose.p);
    camera.position.copy(position);
    camera.quaternion.copy(pose.q);
    angles(pose.q);
    if (openingRail >= 1.8) openingRail = null;
    renderScene();
    return;
  }
  const forward = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)),
    right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  if (tween) {
    tween.t += dt;
    const u = Math.min(1, tween.t / tween.duration),
      s = u * u * (3 - 2 * u);
    position.lerpVectors(tween.start, tween.end, s);
    constrain(position);
    camera.quaternion.slerpQuaternions(tween.q0, tween.q, s);
    angles(camera.quaternion);
    if (u === 1) tween = null;
  } else if (cruise && path) {
    cruiseU = Math.min(
      1,
      cruiseU +
        (config.exactCruise && activeType === "splat"
          ? dt / config.cruiseDuration
          : (dt * (config.cruiseSpeed ?? 0.22)) /
            Math.max(0.01, path.getLength())),
    );
    setCruise(cruiseU);
    if (cruiseU >= 1) {
      cruise = false;
      updateHUD();
    }
  } else {
    yaw +=
      ((keys.has("KeyQ") ? 1 : 0) - (keys.has("KeyE") ? 1 : 0)) * dt * 0.85;
    const x = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0),
      z = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0),
      y = flight
        ? (keys.has("Space") ? 1 : 0) - (keys.has("ShiftLeft") ? 1 : 0)
        : (keys.has("KeyR") ? 1 : 0) - (keys.has("KeyF") ? 1 : 0);
    const input = forward.multiplyScalar(z).addScaledVector(right, x);
    if (input.length() > 1) input.normalize();
    input.y = y;
    if (flight) {
      /* Hover thrust (0,g,0) cancels gravity; increment supplies acceleration. Original implementation. */ velocity.addScaledVector(
        input,
        dt * 1.5,
      );
      velocity.multiplyScalar(Math.exp(-dt * 1.6));
      velocity.clampLength(0, config.speed * 2);
      tilt.rotation.x = THREE.MathUtils.damp(tilt.rotation.x, -z * 0.16, 5, dt);
      tilt.rotation.z = THREE.MathUtils.damp(tilt.rotation.z, -x * 0.16, 5, dt);
    } else {
      input.multiplyScalar(config.speed * (keys.has("ShiftLeft") ? 2.5 : 1));
      velocity.lerp(input, 1 - Math.exp(-dt * 7));
    }
    const wanted = position.clone().addScaledVector(velocity, dt);
    if (!guard && navigation) position.copy(navigation.move(position, wanted, flight ? .22 : .12));
    else position.copy(constrain(wanted.clone()));
    if (wanted.distanceToSquared(position) > 1e-10) velocity.multiplyScalar(.3);
    if (!flight || (strictGS() && guard)) camera.rotation.set(pitch, yaw, roll, "YXZ");
  }
  drone.position.copy(position);
  drone.rotation.y = yaw;
  contact.visible = flight;

  for (const r of rotors) r.rotation.z += dt * 60;
  if (flight && strictGS() && guard) {
    camera.position.copy(position);
  } else if (flight) {
    const behind = new THREE.Vector3(0, .2, .9).applyQuaternion(
      new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ")));

    const desired = constrain(position.clone().add(behind));
    const nextCamera = camera.position.clone().lerp(desired, 1 - Math.exp(-dt * 6));
    camera.position.copy(!guard && navigation ? navigation.move(camera.position, nextCamera) : constrain(nextCamera));
    const target = position
      .clone()
      .add(
        new THREE.Vector3(-Math.sin(yaw) * 0.15, -0.35, -Math.cos(yaw) * 0.15),
      );
    camera.quaternion.slerp(
      quatAt(camera.position, target),
      1 - Math.exp(-dt * 7),
    );
  } else camera.position.copy(position);
  if (strictGS() && guard) {
    if (guard) rail.constrain(camera.position, camera.quaternion, dt, flight);
    position.copy(camera.position);
    angles(camera.quaternion);
    if (flight) {
      const ahead = camera.getWorldDirection(new THREE.Vector3());
      ahead.y = 0;
      ahead.normalize();
      drone.position
        .copy(camera.position)
        .addScaledVector(ahead, config.droneFollowDistance ?? 1.25);
      drone.position.y = camera.position.y + (config.droneOffsetY ?? -0.2);
      drone.rotation.y = yaw;
    }
  }
  // All scene transforms produce Y-up display coordinates. Project the FINAL
  // displayed body, after GS camera constraints and model offsets, onto its floor.
  const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(config.groundY ?? -0.9));
  floor.projectPoint(drone.position, contact.position);
  contact.position.addScaledVector(floor.normal, 0.012); // avoid z-fighting
  contact.scale.setScalar(1 + Math.max(0, floor.distanceToPoint(drone.position)) * 0.2);
  contactLink.visible = flight && (config.groundProjectionLink ?? true);
  const linkPositions = contactLinkGeometry.getAttribute('position');
  linkPositions.setXYZ(0, drone.position.x, Math.max(contact.position.y, drone.position.y - 0.05), drone.position.z);
  linkPositions.setXYZ(1, contact.position.x, contact.position.y, contact.position.z);
  linkPositions.needsUpdate = true;
  contactLinkGeometry.computeBoundingSphere();
  renderScene();
}
window.addEventListener("keydown", (e) => {
  wakeRender();
  if (e.target.closest("select,input")) return;
  if (e.target.closest("button")) e.target.blur();
  if (["Tab", "Space", "ArrowUp", "ArrowDown"].includes(e.code))
    e.preventDefault();
  keys.add(e.code);
  if (e.repeat || !ready) return;
  if (
    [
      "KeyW",
      "KeyA",
      "KeyS",
      "KeyD",
      "KeyR",
      "KeyF",
      "Space",
      "KeyQ",
      "KeyE",
    ].includes(e.code)
  ) {
    tween = null;
    openingRail = null;
    overviewTimer = null;
    cruise = false;
    updateHUD();
  }
  if (e.code === "Tab") toggleFlight();
  if (e.code === "KeyH") document.body.classList.toggle("clean");
  if (e.code === "KeyG") {
    guard = !guard;
    constrain(position);
    updateHUD();
  }
  if (e.code === "KeyT") line.visible = !line.visible;
  if (e.code === "KeyV" && config.pointUrl)
    load(index, activeType === "splat" ? "points" : "splat", true);
  if (e.code === "KeyC" && activeType === "points") {
    heightColors = !heightColors;
    cloud.material.uniforms.heightMode.value = +heightColors;
  }
  if (e.code === "KeyP") {
    openingRail = null;
    overviewTimer = null;
    cruise = !cruise;
    cruiseU = 0;
    tween = null;
    flight = false;
    drone.visible = false;
    updateHUD();
  }
  if (e.code.startsWith("Digit")) {
    const v = config.presets?.[Number(e.code.slice(5)) - 1];
    if (v) go(v);
  }
  if (e.code === "BracketRight" || e.code === "BracketLeft") navigate(configs[(index + (e.code === "BracketRight" ? 1 : configs.length - 1)) % configs.length].id);
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());
let dragging = false;
renderer.domElement.addEventListener("dblclick", () => renderer.domElement.requestPointerLock());
renderer.domElement.addEventListener("pointerdown", (e) => {
  wakeRender(); dragging = true;
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener("pointerup", () => (dragging = false));
renderer.domElement.addEventListener("pointermove", (e) => {
  if ((!dragging && document.pointerLockElement !== renderer.domElement) || !ready) return;
  wakeRender();
  tween = null;
  cruise = false;
  yaw -= e.movementX * 0.003;
  pitch = THREE.MathUtils.clamp(pitch - e.movementY * .003, -85 * Math.PI / 180, 85 * Math.PI / 180);
});
$("hide").onclick = () => document.body.classList.add("clean");
window.addEventListener("pagehide", () => {
  clearTimeout(idlePaintTimer); clearTimeout(resizeTimer); renderer.setAnimationLoop(null); clearRoot();
  spark.defaultView.dispose(); renderer.dispose(); renderer.forceContextLoss();
});
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  const apply = () => {
    const now = performance.now(), ratio = devicePixelRatio * drawingScale;
    if (viewportWidth === innerWidth && viewportHeight === innerHeight && Math.abs(renderer.getPixelRatio() - ratio) < 1e-6) return;
    if (now - lastResizeAt < 500) { resizeTimer = setTimeout(apply, 500 - (now - lastResizeAt)); return; }
    viewportWidth = innerWidth; viewportHeight = innerHeight;
    camera.aspect = innerWidth / innerHeight; projectCamera();
    // Change dimensions AND DPR once, with no CSS/DOM size mutation.
    if (!config?.captureCalibration) renderer.setDrawingBufferSize(innerWidth, innerHeight, ratio);
    noteResize(now);
    wakeRender(); lastMotionAt = now;
  };
  resizeTimer = setTimeout(apply, 150);
});
let slowFrames = [], adaptiveCooldown = performance.now() + 12000;
let lastAnimationTimestamp = null, duplicateAnimationFrames = 0;
function navigate(id, nextQuality = quality, preserve = false) {
  // A new document is the disposal boundary: releases GPU context, Spark's pooled
  // WASM workers and accumulators, unlike SplatMesh.dispose() alone in 0.1.10.
  if (preserve) sessionStorage.setItem("viewer-pose", JSON.stringify({id: config.id,
    position: camera.position.toArray(), quaternion: camera.quaternion.toArray()}));
  const url = new URL(location.href); url.searchParams.set("scene", id);
  url.searchParams.set("quality", nextQuality); location.replace(url);
}
$("quality").onchange = e => navigate(config.id, e.target.value, true);
function animateFrame(timestamp) {
  if (timestamp === lastAnimationTimestamp) duplicateAnimationFrames++;
  lastAnimationTimestamp = timestamp;
  const now = performance.now(),
    dt = Math.min((now - last) / 1000, 0.04);
  const frameMs = now - last; currentFrameMs = frameMs;
  last = now;
  if (ready && !document.hidden && !window.manualCapture && !params.has("benchmark") && now > adaptiveCooldown) {
    slowFrames.push(frameMs);
    const elapsed = slowFrames.reduce((sum, ms) => sum + ms, 0);
    if (!discreteGPU && elapsed >= 1500) {
      const overloaded = elapsed / slowFrames.length > 50;
      slowFrames = []; adaptiveCooldown = now + 1000;
      if (overloaded && quality !== 'low') navigate(config.id, 'low', true);
    } else if (discreteGPU && slowFrames.length >= 90) {
      const slow = slowFrames.filter(ms => ms > 40).length > 60;
      slowFrames = []; adaptiveCooldown = now + 8000;
      if (slow && quality !== 'low') navigate(config.id, quality === 'high' ? 'medium' : 'low', true);
    }
  }
  if (!window.manualCapture) step(dt);
}
renderer.setAnimationLoop(animateFrame);
window.viewer = {
  load: i => navigate(configs[i].id),
  step,
  go,
  toggleFlight,
  referenceView() {
    guard = false; velocity.set(0,0,0); flight = false; drone.visible = false;
    openingRail = null; overviewTimer = null; tween = null; cruise = false;
    position.copy(transform(config.initial.position)); camera.position.copy(position);
    camera.quaternion.copy(config.initial.quaternion ? rail.view(config.initial).q : quatAt(position, transform(config.initial.target)));
    angles(camera.quaternion); renderScene();
  },
  moveBy(offset, yawDegrees = 0) {
    openingRail = null; tween = null; cruise = false; velocity.set(0,0,0);
    const desired = position.clone().add(V(offset));
    position.copy(!guard && navigation ? navigation.move(position, desired, flight ? .22 : .12) : constrain(desired));
    yaw += THREE.MathUtils.degToRad(yawDegrees); camera.rotation.set(pitch, yaw, roll, "YXZ");
    if (!flight) camera.position.copy(position); renderScene();
  },
  get navigation() { return navigation; },
  get captureCenters() { return coverage.map(p => p.toArray()); },
  recordAt(seconds) {
    wakeRender();
    openingRail = null;
    overviewTimer = null;
    if (config.exactCruise && activeType === "splat") {
      this.cruiseAt(seconds / config.cruiseDuration);
      return;
    }
    this.cruiseAt(
      Math.min(
        0.9999,
        (seconds * (config.cruiseSpeed ?? 0.22)) / path.getLength(),
      ),
    );
  },
  cruiseAt(u) {
    tween = null;
    cruise = false;
    setCruise(u);
    camera.position.copy(position);
    renderScene();
  },
  get state() {
    return {
      droneModel: droneModelStatus, droneVisible: drone.visible,
      ready, quality, assetUrl: activeAssetUrl, resolutionScale, drawingScale, renderLoopRunning, idleSettled, duplicateAnimationFrames, resizeCount, resizeWallMs, minResizeIntervalMs, renderFrame: renderer.info.render.frame, drawingBuffer: [renderer.domElement.width, renderer.domElement.height], gpuName, yaw, pitch, collisionHits: navigation?.hits, collisionBounds: navigation?.data.bounds, gaussianCount: cloud?.packedSplats?.numSplats,
      flight,
      cruise,
      guard,
      position: position.toArray(),
      camera: camera.position.toArray(),
      scene: config.id,
      representation: activeType,
      points: Boolean(config.pointUrl),
      intro: overviewTimer !== null || openingRail !== null,
      cameraQuaternion: camera.quaternion.toArray(),
      dronePosition: drone.position.toArray(),
      contactPosition: contact.position.toArray(),
      groundY: config.groundY ?? -0.9,
      droneForward: new THREE.Vector3(0, 0, -1)
        .applyQuaternion(drone.quaternion)
        .toArray(),
      guardReferenceDistance: rail
        ? rail.nearest(camera.position, camera.quaternion).distance
        : null,
      angleErrorDeg: rail
        ? THREE.MathUtils.radToDeg(
            camera.quaternion.angleTo(
              rail.nearest(camera.position, camera.quaternion).q,
            ),
          )
        : null,
      corridorDistance: Math.sqrt(
        Math.min(...coverage.map((p) => p.distanceToSquared(camera.position))),
      ),
    };
  },
};
try {
  // One or more scene manifests, comma-separated; [ / ] or the scene buttons switch between them.
  const manifestUrl = (params.get('manifest') || 'scene.json').split(',')[0];
  configs = await Promise.all((params.get('manifest') || 'scene.json').split(',').map(loadSceneContract));
  configs.forEach((c, i) => { if (configs.findIndex(o => o.id === c.id) !== i) c.id = `${c.id}-${i}`; });
  const model = await loadDroneModel(configs[0].sceneContract.droneModel, manifestUrl);
  tilt.add(model.group);
  rotors.push(...model.rotors);
  droneModelStatus = model.status;
  for (const [id, label, group] of configs.map(s=>[s.id,s.name,s.id])) {
    const b = document.createElement("button"); b.textContent = label; b.dataset.group = group;
    b.onclick = () => navigate(id); $("scenes").append(b);
  }
  if (softwareGPU && !params.has("allowSoftware")) {
    renderer.setAnimationLoop(null);
    $("loading").classList.add("done"); $("error").hidden = false;
    $("error").textContent = "浏览器正在使用软件渲染，已暂停三维加载以避免电脑卡顿。请在 Edge 设置 → 系统和性能中启用图形加速并重启浏览器。独显不是必要条件。";
    window.viewerBlocked = "software-renderer";
  } else {
    const id = params.get("scene") || configs[0].id;
    await load(Math.max(0, configs.findIndex(s => s.id === id)));
  }
  if (params.has("clean")) document.body.classList.add("clean");
} catch (e) {
  console.error(e);
  $("error").hidden = false;
}
