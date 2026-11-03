import * as THREE from 'three';

// Convert the exported calibration/corridor contract, not a scene-name registry.
// Default start: the training view that looks down the longest stretch of the capture path
// (most other camera centres 1-6 m ahead within 25 deg), not views[0], which may face a wall.
function openView(views) {
  let best = views[0], score = -1;
  for (const v of views) {
    const f = v.target.map((t, j) => t - v.position[j]);
    let n = 0;
    for (const w of views) {
      const d = w.position.map((p, j) => p - v.position[j]), r = Math.hypot(...d);
      if (r < 1 || r > 6) continue;
      if ((d[0]*f[0] + d[1]*f[1] + d[2]*f[2]) / r > Math.cos(25 * Math.PI / 180)) n++;
    }
    if (n > score) { score = n; best = v; }
  }
  return best;
}
export async function loadSceneContract(url) {
  const absolute = new URL(url, location.href);
  const response = await fetch(absolute);
  if (!response.ok) throw Error(`Scene manifest: HTTP ${response.status}`);
  const scene = await response.json();
  if (!scene.preferred_model?.full_sh) throw Error('Full SH model required');
  const corridor = scene.capture_corridor;
  const trained = new Set(corridor.training_frame_ids);
  const views = [];
  corridor.frame_ids.forEach((id, i) => {
    if (!trained.has(id)) return;
    const r = corridor.rotations_c2w[i], p = corridor.centers[i];
    // CV camera (right/down/forward) to GL camera (right/up/back).
    const matrix = new THREE.Matrix4().set(
      r[0][0], -r[0][1], -r[0][2], 0,
      r[1][0], -r[1][1], -r[1][2], 0,
      r[2][0], -r[2][1], -r[2][2], 0, 0, 0, 0, 1);
    views.push({position:p, target:p.map((v,j)=>v+r[j][2]),
      quaternion:new THREE.Quaternion().setFromRotationMatrix(matrix).toArray()});
  });
  if (!views.length) throw Error('No training views in capture corridor');
  const model = new URL(scene.preferred_model.path, absolute).href;
  const first = scene.capture_cameras.find(c=>trained.has(c.frame_id));
  // Optional world_up for sources that are not Z-up (e.g. image-only SfM): rotate it onto the viewer's +Y.
  const up = scene.world_up ?? scene.coordinates?.world_up ?? [0, 0, 1];
  const rotation = (up[0] === 0 && up[1] === 0 && up[2] > 0) ? null
    : new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(...up).normalize(), new THREE.Vector3(0, 1, 0)).toArray();
  const toViewer = p => rotation ? new THREE.Vector3(...p).applyQuaternion(new THREE.Quaternion().fromArray(rotation))
    : new THREE.Vector3(p[0], p[2], -p[1]);
  // Scene bounds are reconstruction extents; margin is in exported scene units.
  // Legacy manifests without bounds keep the camera-centre + 1 m fallback.
  const rawMargin = new URLSearchParams(location.search).get('margin');
  const parsedMargin = rawMargin === null ? 1.5 : Number(rawMargin);
  const margin = Number.isFinite(parsedMargin) && parsedMargin >= 0 ? parsedMargin : 1.5;
  const box = scene.bounds
    ? new THREE.Box3().setFromPoints([0,1,2,3,4,5,6,7].map(i => toViewer([0,1,2].map(j =>
        (i >> j) & 1 ? scene.bounds.max[j] : scene.bounds.min[j])))).expandByScalar(margin)
    : new THREE.Box3().setFromPoints(views.map(v => toViewer(v.position))).expandByScalar(1);
  return {id:scene.id ?? 'scene', name:scene.name || '重建场景', subtitle:scene.subtitle, type:'splat', up:'Z',
    ...(rotation ? {rotation} : {}),
    url:model, ...(scene.cloud ? {pointUrl:new URL(scene.cloud,absolute).href} : {}),
    highQualityUrl:scene.web_assets?.high ? new URL(scene.web_assets.high,absolute).href : model,
    mediumUrl:scene.web_assets?.medium ? new URL(scene.web_assets.medium,absolute).href : model,
    lowUrl:scene.web_assets?.low ? new URL(scene.web_assets.low,absolute).href : model,
    initial:scene.initial_view ?? openView(views), speed:.3, gsStrict:true,
    corridorRadius:corridor.recommended_max_distance,
    angleLimitDeg:corridor.recommended_max_angle_deg,
    captureCalibration:first,
    trajectoryData:{path:views, coverage:views.map(v=>v.position),coverageViews:views,fps:30},
    navigationData:{bounds:{min:box.min.toArray(),max:box.max.toArray()},
      occupied:[], origin:box.min.toArray(),dims:[1,1,1],cellSize:.15},
    sceneContract:scene};
}
