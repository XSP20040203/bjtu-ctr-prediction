const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname,source=JSON.parse(fs.readFileSync(path.join(root,'model_data.json'),'utf8')),model=JSON.parse(fs.readFileSync(path.join(root,'integrated_model.netron'),'utf8'));
const checks=[],errors=[],external=[];let testBrowser;function check(ok,label,detail){checks.push({label,pass:!!ok,detail});if(!ok)throw Error(label+': '+JSON.stringify(detail));}
const main=model.graphs[0];const nodeByOut=new Map(main.nodes.flatMap(n=>n.outputs.flatMap(o=>o.value.map(v=>[main.values[v].name,n]))));
function parents(key){return nodeByOut.get(key).inputs.flatMap(i=>i.value.map(v=>main.values[v].name));}
check(parents('day_binary')[0]==='calibrated'&&Array.from({length:12},(_,i)=>parents(`opinion_${i+1}`)[0]).every(p=>p==='calibrated'),'用户日两路与十二意见共用校准起点');
check(JSON.stringify(parents('noncold_context'))===JSON.stringify(parents('cold_context')),'冷/非冷395维树同父');
check(['features533','features489','features401'].every(k=>JSON.stringify(parents(k))===JSON.stringify(['frozen_teacher','context_features395'])),'三种表示输入独立组装');
for(let i=0;i<source.sections.length;i++){const s=source.sections[i],g=model.graphs[i+1];check(g.name===s.title,s.id+' 图名完整');check(g.nodes.length===s.nodes.length,s.id+' 节点覆盖');check(g.nodes.every((n,j)=>n.name===s.nodes[j].title&&n.attributes.map(a=>a.value).join('\n')===s.nodes[j].body),s.id+' 结构正文逐字相同');const producers=new Map(g.nodes.flatMap((n,j)=>n.outputs.flatMap(o=>o.value.map(v=>[v,j]))));const edges=g.nodes.flatMap((n,j)=>n.inputs.flatMap(k=>k.value.map(v=>[producers.get(v),j])));check(JSON.stringify(edges.slice().sort())===JSON.stringify(s.edges.slice().sort()),s.id+' 连接关系相同');}
(async()=>{
 const browser=await chromium.launch({headless:true});testBrowser=browser;const context=await browser.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:1});const page=await context.newPage();
 page.on('pageerror',e=>errors.push(e.stack));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('requestfailed',r=>errors.push(r.url()+': '+r.failure()?.errorText));page.on('response',r=>{if(r.status()>=400)errors.push(r.url()+': '+r.status())});page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:8765/')&&!r.url().startsWith('blob:')&&!r.url().startsWith('data:'))external.push(r.url())});
 await page.goto('http://127.0.0.1:8765/index.html?view=netron');await page.waitForFunction(()=>MODEL_EXPLORER.ready,null,{timeout:20000});
 const frame=page.frameLocator('#viewer');
 check(await page.locator('#graph-select option').count()===36,'36个命名图可以选择');
 check(await page.evaluate(()=>MODEL_EXPLORER.app.model.modules[0].nodes.length)===88,'实际Netron打开88节点完整图');
 check(await page.evaluate(()=>document.getElementById('loading').hidden),'加载提示结束');
 await page.screenshot({path:path.join(root,'qa','netron-desktop.png')});
 const zoom=await page.evaluate(()=>MODEL_EXPLORER.app.target.zoom);await page.locator('#zoom-in').click();check(await page.evaluate(()=>MODEL_EXPLORER.app.target.zoom)>zoom,'放大按钮');await page.locator('#zoom-out').click();
 // Use outer-world evaluation: Netron deliberately disables eval in its window.
 let target=await page.evaluate(()=>{const d=document.getElementById('viewer').contentDocument;const t=d.getElementById('target');return {left:t.scrollLeft,top:t.scrollTop}});
 await page.mouse.move(800,650);await page.mouse.wheel(0,-120);await page.waitForTimeout(150);
 check(await page.evaluate(()=>MODEL_EXPLORER.app.target.zoom)>zoom,'滚轮缩放');
 await page.mouse.move(1300,850);await page.mouse.down();await page.mouse.move(1100,600,{steps:8});await page.mouse.up();
 let moved=await page.evaluate(()=>{const t=document.getElementById('viewer').contentDocument.getElementById('target');return {left:t.scrollLeft,top:t.scrollTop}});
 check(moved.left!==target.left||moved.top!==target.top,'画布拖动平移',{target,moved});
 await page.locator('#fit').click();check(await page.evaluate(()=>MODEL_EXPLORER.app.target.zoom)<zoom,'适应完整图');await page.locator('#start').click();
 await page.locator('#properties').click();check((await frame.locator('#sidebar').innerText()).includes('2019-11-17'),'模型属性显示预测日');await frame.locator('#sidebar-closebutton').click();
 await page.evaluate(()=>MODEL_EXPLORER.choose(14)); // Actual parallel user-day graph.
 await page.waitForFunction(()=>MODEL_EXPLORER.app.activeTarget.name==='用户日两条并行路径总图');
 check(await frame.locator('g.node').count()===12,'用户日并行详图12节点');
 await page.locator('#fit').click();await page.screenshot({path:path.join(root,'qa','netron-parallel.png')});
 const title=frame.locator('text').filter({hasText:/^十二意见逻辑元模型$/}).first();await title.click();
 check((await frame.locator('#sidebar').innerText()).includes('0.01'),'点击节点在右侧显示真实属性');
 await page.screenshot({path:path.join(root,'qa','netron-properties.png')});await frame.locator('#sidebar-closebutton').click();
 for(let i=1;i<model.graphs.length;i++){
  await page.evaluate(i=>MODEL_EXPLORER.choose(i),i);
  const current=await page.evaluate(()=>({name:MODEL_EXPLORER.app.activeTarget.name,count:MODEL_EXPLORER.app.activeTarget.nodes.length,visible:document.getElementById('viewer').contentDocument.querySelectorAll('g.node').length,body:document.getElementById('viewer').contentDocument.body.className}));
  check(current.name===model.graphs[i].name&&current.count===model.graphs[i].nodes.length&&current.visible===current.count&&!current.body.includes('alert'),'Netron渲染详图 '+i,current);
 }
 // Drill into a native Netron function using its ƒ glyph, then return.
 await page.evaluate(()=>MODEL_EXPLORER.choose(0));await page.locator('#fit').click();
 const node=frame.locator('g.node').filter({hasText:'用户日期对齐与原始行序保留'}).first();
 await node.locator('text').filter({hasText:/^ƒ$/}).click();await page.waitForFunction(()=>MODEL_EXPLORER.app.activeTarget.name==='用户日期对齐与原始行序保留');
 check(await page.evaluate(()=>MODEL_EXPLORER.app.activeTarget.nodes.length)===7,'原生ƒ进入内部结构');
 await frame.locator('#toolbar-path-back-button').click();await page.waitForFunction(()=>MODEL_EXPLORER.app.activeTarget.name.startsWith('完整整合模型'));check(true,'原生返回恢复完整模型');
 await page.locator('#find').click();await frame.locator('#sidebar input').first().fill('401');await page.waitForTimeout(100);check((await frame.locator('#sidebar').innerText()).includes('历史点积'),'原生搜索支持特征维度');await frame.locator('#sidebar-closebutton').click();
 await page.evaluate(()=>MODEL_EXPLORER.choose(30));await page.locator('#fit').click();
 const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadPromise;await download.saveAs(path.join(root,'qa','netron-三种表示输入.svg'));check(fs.statSync(path.join(root,'qa','netron-三种表示输入.svg')).size>1000,'导出SVG可用');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
 check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'手机工具栏无页面横向溢出');
 await page.evaluate(()=>MODEL_EXPLORER.choose(30));await page.locator('#fit').click();await page.screenshot({path:path.join(root,'qa','netron-mobile.png')});
 await page.locator('#help').click();check(await page.locator('#help-dialog').isVisible(),'手机帮助可打开');await page.locator('#help-close').click();
 await page.reload();await page.waitForFunction(()=>MODEL_EXPLORER.ready&&MODEL_EXPLORER.app.activeTarget.nodes.length===5,null,{timeout:20000});check(true,'刷新保留模块选择');
 check(errors.length===0,'无脚本与资源错误',errors);check(external.length===0,'查看器资源与结构均在本地加载',external);
 fs.writeFileSync(path.join(root,'qa','netron-validation.json'),JSON.stringify({checkedAt:new Date().toISOString(),checks,errors,external,summary:{passed:checks.length,graphs:36,mainNodes:88,detailNodes:204}},null,2));
 console.log(JSON.stringify({passed:checks.length,graphs:36,mainNodes:88,detailNodes:204,errors,external}));await browser.close();
})().catch(async e=>{fs.writeFileSync(path.join(root,'qa','netron-validation-failure.json'),JSON.stringify({error:e.stack,checks,errors,external},null,2));console.error(e);await testBrowser?.close();process.exitCode=1;});
