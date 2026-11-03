// Headful functional acceptance; reports the actual WebGL GPU, not FPS.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {chromium}=require('playwright-core');
const root=path.resolve(process.argv[2]||'.'),manifest=path.resolve(process.argv[3]||'scene.json');
const report=path.resolve(process.argv[4]||'viewer-result.json');
const extra=new URLSearchParams(process.argv[5]||'');
const args=JSON.parse(process.env.CHROME_ARGS||'[]');
const mime={'.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json',
  '.glb':'model/gltf-binary','.png':'image/png','.wasm':'application/wasm'};
(async()=>{
 let browser;const errors=[],modelResponses=[];let result={passed:false,headless:false,args};
 const server=http.createServer((req,res)=>{
   const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
   if(pathname==='/favicon.ico'){res.writeHead(204).end();return;}
   const requested=path.resolve(root,'.'+pathname);
   if(!requested.startsWith(root+path.sep)||!fs.existsSync(requested)||!fs.statSync(requested).isFile()) {res.writeHead(404).end();return;}
   res.setHeader('Content-Type',mime[path.extname(requested)]||'application/octet-stream');
   res.setHeader('Content-Length',fs.statSync(requested).size);fs.createReadStream(requested).pipe(res);
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try {
   browser=await chromium.launch({executablePath:process.env.CHROME||'/usr/bin/microsoft-edge',headless:false,args});
   const page=await browser.newPage({viewport:{width:1280,height:720}});
   page.on('pageerror',e=>errors.push(String(e)));
   page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
   page.on('response',r=>{if(new URL(r.url()).pathname.endsWith('.glb'))modelResponses.push({url:r.url(),status:r.status()});});
   const rel=path.relative(root,manifest).split(path.sep).join('/');
   if(rel.startsWith('../'))throw Error('Manifest must be under served root');
   const query=new URLSearchParams({manifest:'/'+rel,quality:'high',capture:'1',benchmark:'1'});
   for(const [k,v] of extra)query.set(k,v);
   const url=`http://127.0.0.1:${server.address().port}/viewer/index.html?${query}`;
   await page.goto(url);
   await page.waitForFunction(()=>window.viewerReady===true,{},{timeout:180000});
   await page.evaluate(()=>window.viewer.referenceView());
   await page.waitForFunction(()=>window.viewer.state.idleSettled,{},{timeout:120000});
   const state=await page.evaluate(()=>({device:window.viewerDevice,state:window.viewer.state}));
   const screenshot=report.replace(/\.json$/,'.png');
   await page.screenshot({path:screenshot});
   await page.keyboard.press('Tab');
   await page.waitForFunction(()=>window.viewer.state.flight && window.viewer.state.droneVisible);
   await page.waitForTimeout(500);
   const flightScreenshot=report.replace(/\.json$/,'-drone.png');
   await page.screenshot({path:flightScreenshot});
   const flightState=await page.evaluate(()=>window.viewer.state);
   const drone=state.state.droneModel;
   const loaded=drone?.loaded&&drone.meshCount>0&&(drone.url==='built-in quadrotor'||modelResponses.some(r=>r.url===drone.url&&r.status===200));
   result={...result,passed:errors.length===0&&!state.device.software&&state.state.gaussianCount>0&&!!loaded,
     errors,url,...state,flightState,modelResponses,screenshot,flightScreenshot};
   if(!result.passed)throw Error('Viewer acceptance failed');
 }catch(e){result={...result,passed:false,error:String(e),errors};process.exitCode=1;}
 finally {
   fs.writeFileSync(report,JSON.stringify(result,null,2)+'\n');
   console.log(JSON.stringify(result,null,2));
   await browser?.close();await new Promise(resolve=>server.close(resolve));
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
