const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname,model=JSON.parse(fs.readFileSync(path.join(root,'..','integrated_model.netron'),'utf8')),source=model.graphs[0];
const manifest=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
const checks=[],errors=[];function check(ok,name,detail){checks.push({name,pass:!!ok,detail});if(!ok)throw Error(name+': '+JSON.stringify(detail));}
const squash=s=>s.replace(/\s/g,'');
let browser;
(async()=>{
 browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('file:///'+root.replace(/\\/g,'/')+'/index.html',{waitUntil:'load'});
 await page.waitForFunction(()=>window.FLOWCHART_VIEWER?.selected==='full');
 check(await page.locator('#surface g.node').count()===93,'完整图88计算节点、4输入、1输出全部存在');
 check(await page.locator('#surface g.edge').count()===manifest.edges,'完整图所有146条连接全部存在');
 const content=await page.evaluate(()=>[...document.querySelectorAll('#surface g.node')].map(g=>({id:g.id,text:[...g.querySelectorAll('text')].map(t=>t.textContent).join(' ')})));
 source.nodes.forEach((n,i)=>{
   const t=squash(content.find(c=>c.id==='node-'+i).text);
   check(t.includes(squash(n.name))&&n.attributes.every(a=>t.includes(squash(a.name+'：'+a.value))),'模块名称、参数与维度完整 '+(i+1));
 });
 const expected=new Set(),producer=new Map();source.inputs.forEach((p,i)=>p.value.forEach(v=>producer.set(v,'in'+i)));source.nodes.forEach((n,i)=>n.outputs.forEach(p=>p.value.forEach(v=>producer.set(v,'n'+i))));source.nodes.forEach((n,i)=>n.inputs.forEach(p=>p.value.forEach(v=>expected.add(producer.get(v)+'->n'+i))));expected.add('n87->out');
 const got=new Set(await page.locator('#surface g.edge title').allTextContents());
 check([...expected].every(e=>got.has(e))&&got.size===expected.size,'每一条连接与原结构逐条相同');
 check(got.has('n21->n22')&&Array.from({length:12},(_,i)=>got.has('n21->n'+(26+i))).every(Boolean),'用户日两路与十二意见从共同起点并行');
 check(got.has('n73->n74')&&got.has('n73->n75'),'冷任务与非冷任务共用同一个父结果');
 check([80,81,82].every(i=>got.has('n78->n'+i)&&got.has('n79->n'+i)),'533、489、401三份输入独立组装');
 const ids=['full',...manifest.stages.map(s=>s.id),...manifest.details.map(s=>'detail:'+s.id)];
 for(const id of ids){
   await page.evaluate(id=>FLOWCHART_VIEWER.choose(id),id);
   await page.waitForTimeout(20);
   // Measure at 1:1; Chromium rounds glyph extents at tiny overview scales.
   await page.evaluate(()=>FLOWCHART_VIEWER.zoom(1));
   const bad=await page.evaluate(()=>[...document.querySelectorAll('#surface g.node')].flatMap(g=>{
     const polys=[...g.querySelectorAll('polygon')];if(!polys.length)return [];
     const box=polys.map(p=>p.getBBox()).reduce((a,b)=>({x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),r:Math.max(a.r,b.x+b.width),b:Math.max(a.b,b.y+b.height)}),{x:Infinity,y:Infinity,r:-Infinity,b:-Infinity});
     return [...g.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<box.x-2||b.x+b.width>box.r+2||b.y<box.y-2||b.y+b.height>box.b+2}).map(t=>{const b=t.getBBox();return {node:g.id,text:t.textContent,bounds:{x:b.x,y:b.y,width:b.width,height:b.height},box};});
   }));
   check(bad.length===0,'中文文字均在框内 '+id,bad.slice(0,12));
 }
 await page.evaluate(()=>FLOWCHART_VIEWER.choose('day'));await page.waitForTimeout(80);
 await page.screenshot({path:path.join(root,'qa','用户日并行流程.png'),timeout:60000});
 const scale=await page.evaluate(()=>FLOWCHART_VIEWER.scale);await page.locator('#zoom-in').click();check(await page.evaluate(()=>FLOWCHART_VIEWER.scale)>scale,'放大控件');await page.locator('#fit').click();
 await page.evaluate(()=>FLOWCHART_VIEWER.choose('representation'));await page.waitForTimeout(80);await page.screenshot({path:path.join(root,'qa','最终表示与输出.png'),timeout:60000});
 await page.locator('#node-86').click();check(await page.locator('#inspector').isVisible()&&(await page.locator('#node-attrs').innerText()).includes('401'),'实际点击模块可读完整参数');await page.locator('#open-detail').click();check(await page.evaluate(()=>FLOWCHART_VIEWER.selected.startsWith('detail:')),'内部流程图打开');
 await page.waitForTimeout(80);await page.screenshot({path:path.join(root,'qa','内部结构示例.png'),timeout:60000});
 await page.evaluate(()=>FLOWCHART_VIEWER.choose('full'));await page.waitForTimeout(80);await page.screenshot({path:path.join(root,'qa','完整流程网页.png'),timeout:60000});
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(80);await page.evaluate(()=>FLOWCHART_VIEWER.fit());check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'手机页面无横向溢出');await page.screenshot({path:path.join(root,'qa','手机阅读.png'),timeout:60000});
 check(errors.length===0,'无浏览器脚本错误',errors);
 for(const item of [manifest.full,...manifest.stages,...manifest.details]){check(fs.statSync(path.join(root,item.file)).size>1000,'SVG文件存在 '+item.file);check(fs.statSync(path.join(root,item.file.replace(/\.svg$/,'.png'))).size>1000,'PNG文件存在 '+item.file);}
 fs.writeFileSync(path.join(root,'qa','validation.json'),JSON.stringify({at:new Date().toISOString(),passed:checks.length,checks,errors},null,2));
 console.log(JSON.stringify({passed:checks.length,graphs:ids.length,nodes:88,edges:manifest.edges,errors}));await browser.close();
})().catch(async e=>{fs.writeFileSync(path.join(root,'qa','validation-failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2));console.error(e);await browser?.close();process.exitCode=1;});
