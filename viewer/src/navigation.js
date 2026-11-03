import * as THREE from 'three';
// Coarse reconstructed-surface collision: 15cm voxels, already dilated once.
// It is not a watertight mesh, and cannot guarantee collision in unobserved space.
export class Navigation {
  constructor(data) {
    this.data=data;
    this.box=new THREE.Box3(new THREE.Vector3(...data.bounds.min),new THREE.Vector3(...data.bounds.max));
    this.cells=new Set(data.occupied);
    this.hits=0;
  }
  clamp(p,radius=.12) {
    const lo=this.box.min.clone().addScalar(radius), hi=this.box.max.clone().addScalar(-radius);
    return p.clamp(lo,hi);
  }
  occupied(p,radius=.12) {
    const d=this.data, extra=Math.max(0,radius-d.cellSize), cell=d.cellSize;
    const min=p.toArray().map((v,i)=>Math.floor((v-extra-d.origin[i])/cell));
    const max=p.toArray().map((v,i)=>Math.floor((v+extra-d.origin[i])/cell));
    for(let z=min[2];z<=max[2];z++)for(let y=min[1];y<=max[1];y++)for(let x=min[0];x<=max[0];x++) {
      if(x<0||y<0||z<0||x>=d.dims[0]||y>=d.dims[1]||z>=d.dims[2])continue;
      if(this.cells.has(x+d.dims[0]*(y+d.dims[1]*z))) return true;
    }
    return false;
  }
  move(from, to, radius=.12) {
    const target=this.clamp(to.clone(),radius), p=this.clamp(from.clone(),radius);
    const delta=target.clone().sub(p), n=Math.max(1,Math.ceil(delta.length()/.05));delta.divideScalar(n);
    for(let k=0;k<n;k++)for(const axis of ['x','y','z']) {
      const next=p.clone();next[axis]+=delta[axis];this.clamp(next,radius);
      if(!this.occupied(next,radius))p.copy(next);else this.hits++;
    }
    return p;
  }
}
