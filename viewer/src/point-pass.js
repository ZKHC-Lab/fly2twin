import * as THREE from "three";

// Small original EDL-style depth-contrast pass, restricted to point representation.
// No third-party postprocessing code or additional network dependencies.
export class PointPass {
  constructor(renderer) {
    this.renderer = renderer;
    this.target = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true });
    this.target.depthTexture = new THREE.DepthTexture(
      1,
      1,
      THREE.UnsignedIntType,
    );
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.material = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        image: { value: this.target.texture },
        depth: { value: this.target.depthTexture },
        pixel: { value: new THREE.Vector2() },
        near: { value: 0.03 },
        far: { value: 200 },
      },
      vertexShader:
        "varying vec2 uv0;void main(){uv0=uv;gl_Position=vec4(position.xy,0.,1.);}",
      fragmentShader: `
        uniform sampler2D image,depth;
        uniform vec2 pixel;
        uniform float near,far;
        varying vec2 uv0;
        float linearDepth(float z){return (near*far)/(far-z*(far-near));}
        float difference(vec2 offset,float center){
          float z=texture2D(depth,uv0+offset*pixel).r;
          if(z>.99999)return 0.;
          return max(0.,(center-linearDepth(z))/max(center,.1));
        }
        void main(){
          vec4 color=texture2D(image,uv0);
          float z=texture2D(depth,uv0).r;
          if(z<.99999){
            float d=linearDepth(z);
            float response=difference(vec2(1.6,0.),d)+difference(vec2(-1.6,0.),d)+difference(vec2(0.,1.6),d)+difference(vec2(0.,-1.6),d);
            color.rgb*=exp(-min(response,1.)*1.5);
          }
          gl_FragColor=color;
          #include <colorspace_fragment>
        }`,
    });
    this.scene.add(
      new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material),
    );
  }
  render(scene, camera) {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    if (this.target.width !== size.x || this.target.height !== size.y)
      this.target.setSize(size.x, size.y);
    this.material.uniforms.pixel.value.set(1 / size.x, 1 / size.y);
    this.material.uniforms.near.value = camera.near;
    this.material.uniforms.far.value = camera.far;
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(scene, camera);
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }
}
