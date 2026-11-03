import * as THREE from "three";
export class CaptureRail {
  constructor(data, config, transform) {
    this.config = config;
    const basis = config.rotation
      ? new THREE.Quaternion().fromArray(config.rotation)
      : new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(1, 0, 0),
          config.up === "Z" ? -Math.PI / 2 : 0,
        );
    this.view = (v) => ({
      p: transform(v.position),
      q: v.quaternion
        ? basis.clone().multiply(new THREE.Quaternion().fromArray(v.quaternion))
        : new THREE.Quaternion().setFromRotationMatrix(
            new THREE.Matrix4().lookAt(
              transform(v.position),
              transform(v.target),
              new THREE.Vector3(0, 1, 0),
            ),
          ),
    });
    this.path = data.path.map(this.view);
    this.views = (data.coverageViews ?? data.path).map(this.view);
    this.fps = data.fps ?? 30;
  }
  nearest(p, orientation = null) {
    let i = 0,
      d = Infinity;
    this.views.forEach((v, j) => {
      const n = v.p.distanceToSquared(p);
      if (n < d) {
        d = n;
        i = j;
      }
    });
    if (orientation) {
      // Revisited capture locations can face opposite directions. Resolve the
      // local ambiguity without moving the camera outside its coverage radius.
      const radius = this.config.corridorRadius ?? 0.15;
      const limit = THREE.MathUtils.degToRad(this.config.angleLimitDeg ?? 15);
      const neighborhood = Math.max(radius * 1.2, Math.sqrt(d) + radius * 0.5);
      let score = Infinity;
      this.views.forEach((v, j) => {
        const n = v.p.distanceToSquared(p);
        if (n > neighborhood * neighborhood) return;
        const a = orientation.angleTo(v.q);
        const cost = n + 0.25 * radius * radius * (a / limit) ** 2;
        if (cost < score) {
          score = cost;
          i = j;
        }
      });
      d = this.views[i].p.distanceToSquared(p);
    }
    return { ...this.views[i], distance: Math.sqrt(d), index: i };
  }
  sample(seconds) {
    const f = THREE.MathUtils.clamp(
        seconds * this.fps,
        0,
        this.path.length - 1,
      ),
      i = Math.floor(f),
      a = this.path[i],
      b = this.path[Math.min(i + 1, this.path.length - 1)];
    return {
      p: a.p.clone().lerp(b.p, f - i),
      q: a.q.clone().slerp(b.q, f - i),
    };
  }
  constrain(p, q, dt = 0, followHeight = false) {
    const v = this.nearest(p, q),
      radius = this.config.corridorRadius ?? 0.15,
      limit = THREE.MathUtils.degToRad(this.config.angleLimitDeg ?? 15);
    // Soft spring inside the outer boundary, followed by an exact cap.
    if (followHeight) p.y = v.p.y;
    const delta = p.clone().sub(v.p),
      distance = delta.length();
    if (distance > radius * 0.8 && dt > 0)
      p.lerp(
        v.p.clone().add(delta.setLength(radius * 0.8)),
        (1 - Math.exp(-dt * 7)) * 0.7,
      );
    if (p.distanceTo(v.p) > radius) p.sub(v.p).setLength(radius).add(v.p);
    const angle = q.angleTo(v.q);
    if (angle > limit * 0.8 && dt > 0)
      q.slerp(v.q, (1 - Math.exp(-dt * 8)) * 0.5);
    const remaining = q.angleTo(v.q);
    if (remaining > limit) q.copy(v.q.clone().slerp(q, limit / remaining));
    return {
      distance: p.distanceTo(v.p),
      angleDeg: THREE.MathUtils.radToDeg(q.angleTo(v.q)),
      index: v.index,
    };
  }
}
