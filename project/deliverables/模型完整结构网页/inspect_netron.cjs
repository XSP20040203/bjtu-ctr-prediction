const fs=require('fs');const path=require('path');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1600,height:1000}});const logs=[];const req=[];
page.on('pageerror',e=>logs.push(e.stack));page.on('console',m=>{if(m.type()==='error')logs.push(m.text())});page.on('requestfailed',r=>logs.push(r.url()+':'+r.failure()?.errorText));page.on('request',r=>req.push(r.url()));
await page.goto('http://127.0.0.1:8765/index.html?view=netron');
try{await page.waitForFunction(()=>window.MODEL_EXPLORER?.ready||document.getElementById('viewer').contentDocument?.body?.classList.contains('alert'),null,{timeout:20000});}catch(e){logs.push(e.message)}
await page.screenshot({path:path.join(__dirname,'qa','netron-first.png')});
const state=await page.evaluate(()=>({ready:MODEL_EXPLORER.ready,loading:document.getElementById('loading-text')?.textContent,model:MODEL_EXPLORER.app?.model?.modules?.length,target:MODEL_EXPLORER.app?.target?{width:MODEL_EXPLORER.app.target._width,height:MODEL_EXPLORER.app.target._height,zoom:MODEL_EXPLORER.app.target.zoom}:null}));
const fr=page.frames().find(f=>f.url().includes('vendor/netron'));
if(fr){state.frame=await page.evaluate(()=>{const d=document.getElementById('viewer').contentDocument;return {body:d.body.className,text:d.body.innerText.slice(-3000),nodes:d.querySelectorAll('g.node').length,scripts:[...d.scripts].map(s=>s.src)}});}
console.log(JSON.stringify({state,logs,external:req.filter(x=>!x.startsWith('http://127.0.0.1:8765/')&&!x.startsWith('blob:'))},null,2));await browser.close();
})();
