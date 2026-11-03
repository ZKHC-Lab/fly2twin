import {Vector3, Quaternion} from '../vendor/three/build/three.module.js';
const linear = c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4;
self.onmessage = async ({data}) => {
 try {
  const response=await fetch(data.url);if(!response.ok)throw Error(`PLY HTTP ${response.status}`);
  const bytes=await response.arrayBuffer(), view=new DataView(bytes);
  const header=new TextDecoder().decode(new Uint8Array(bytes,0,Math.min(bytes.byteLength,65536)));
  const end=header.indexOf('end_header\n');if(end<0)throw Error('PLY header missing');
  const text=header.slice(0,end), count=Number(text.match(/element vertex (\d+)/)?.[1]);
  if(!text.includes('format binary_little_endian')||!count)throw Error('Point mode requires binary little-endian PLY');
  const props=[];let stride=0,inVertex=false;
  const types={float:['getFloat32',4],float32:['getFloat32',4],double:['getFloat64',8],uchar:['getUint8',1],uint8:['getUint8',1],int:['getInt32',4]};
  for(const line of text.split('\n')){
    if(line.startsWith('element ')){inVertex=line.startsWith('element vertex ');continue;}
    if(inVertex&&line.startsWith('property ')){const [,type,name]=line.split(/\s+/);if(!types[type])throw Error('Unsupported point PLY property');props.push({name,offset:stride,method:types[type][0]});stride+=types[type][1];}
  }
  const map=Object.fromEntries(props.map(p=>[p.name,p]));
  for(const name of ['x','y','z','red','green','blue'])if(!map[name])throw Error(`Missing ${name}`);
  const positions=new Float32Array(count*3), colors=new Float32Array(count*3);
  const q=data.rotation?new Quaternion(...data.rotation):data.up==='Z'?new Quaternion().setFromAxisAngle(new Vector3(1,0,0),-Math.PI/2):new Quaternion();
  const p=new Vector3(), shift=new Vector3(...(data.translation||[0,0,0]));
  const start=end+'end_header\n'.length;
  for(let i=0;i<count;i++){
    const base=start+i*stride, get=k=>view[map[k].method](base+map[k].offset,true);
    p.set(get('x'),get('y'),get('z')).applyQuaternion(q).multiplyScalar(data.scale??1).add(shift);p.toArray(positions,i*3);
    for(const [j,k] of ['red','green','blue'].entries())colors[i*3+j]=linear(get(k)/255);
  }
  self.postMessage({positions,colors},[positions.buffer,colors.buffer]);
 } catch(e){self.postMessage({error:String(e)})}
};
