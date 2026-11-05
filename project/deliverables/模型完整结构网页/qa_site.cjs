const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname;
const data=JSON.parse(fs.readFileSync(path.join(root,'model_data.json'),'utf8'));
const checks=[];const errors=[];
function assert(condition,label,detail){checks.push({label,pass:!!condition,detail});if(!condition)throw Error(label+': '+JSON.stringify(detail));}
async function overflow(page,label){const result=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,bad:[...document.querySelectorAll('.graph-node,.mini-card,.chain-node')].filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>({text:n.innerText.slice(0,45),client:n.clientWidth,scroll:n.scrollWidth}))}));assert(result.scroll<=result.width+1,label+' 页面无横向溢出',result);assert(!result.bad.length,label+' 节点文字无横向溢出',result.bad);}
(async()=>{
 fs.mkdirSync(path.join(root,'qa'),{recursive:true});
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 const page=await context.newPage();
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 const requests=[];page.on('request',r=>requests.push(r.url()));
 await page.goto(pathToFileURL(path.join(root,'说明阅读.html')).href);
 await page.locator('.stage').last().waitFor();
 assert(await page.locator('.stage').count()===8,'八阶段总览完整');
 assert(await page.locator('#nav .nav-item').count()===35,'导航可达全部35张图');
 await overflow(page,'桌面总览');
 await page.screenshot({path:path.join(root,'qa','desktop-overview.png')});
 await page.locator('#collapse-stages').click();assert(await page.locator('.stage-body:visible').count()===0,'折叠主干');
 await page.locator('#expand-stages').click();assert(await page.locator('.stage-body:visible').count()===8,'展开全部阶段');
 for(const s of data.sections){
  await page.evaluate(id=>setView('module',id),s.id);
  await page.waitForFunction(n=>document.querySelectorAll('#main-view .graph-path').length===n,s.edges.length);
  assert(await page.locator('#main-view .graph-node').count()===s.nodes.length,s.id+' 全部节点可见');
  const bodies=await page.locator('#main-view .node-body').allTextContents();
  assert(bodies.every((b,i)=>b===s.nodes[i].body),s.id+' 维度与参数逐字一致');
  const paragraphs=await page.locator('#main-view .explain p').allTextContents();
  assert(JSON.stringify(paragraphs)===JSON.stringify(s.paragraphs),s.id+' 解释逐字一致');
  await overflow(page,'桌面 '+s.id);
  const geometry=await page.evaluate(()=>[...document.querySelectorAll('.graph-path')].map(p=>p.getAttribute('d')));
  assert(geometry.every(d=>!d.includes('NaN')),s.id+' 连线坐标有效');
 }
 await page.evaluate(()=>setView('module','parallel_day'));await page.waitForTimeout(100);
 await page.screenshot({path:path.join(root,'qa','desktop-parallel.png'),fullPage:true});
 await page.locator('.graph-node').nth(4).click();assert(await page.locator('#node-dialog').isVisible(),'节点详情可以打开');
 assert((await page.locator('#dialog-body').innerText()).includes('岭参数0.01'),'节点详情保留真实参数');
 await page.keyboard.press('Escape');assert(!await page.locator('#node-dialog').isVisible(),'键盘 Escape 关闭详情');
 for(const q of ['1017','312 + 500','[','0.025']){await page.locator('#global-search').fill(q);await page.waitForTimeout(30);assert(await page.locator('#main-view h2').innerText()===`搜索“${q}”`,'搜索 '+q+' 可用');}
 await page.locator('#global-search').fill('1017');assert(await page.locator('.search-result').count()>0,'全文搜索发现维度');
 await page.locator('.search-result').first().click();assert(await page.locator('#global-search').inputValue()==='','选中搜索结果清空搜索');
 await page.evaluate(()=>setView('dimensions'));await page.locator('#dimension-search').fill('540');assert(await page.locator('[data-dim-row]:visible').count()>=1,'维度表筛选');
 await page.evaluate(()=>setView('walk'));
 let result=await page.locator('.result-number').innerText();assert(Math.abs(Number(result)-1/(1+Math.exp(-(.9*Math.log(.02/.98)+.1*-3))))<1e-7,'假设样本最后融合计算准确',result);
 await page.locator('#walk-exposure').fill('101');assert(await page.locator('.result-number').innerText()==='0.0200000','超过100次保留父概率');
 await page.locator('#walk-exposure').fill('100');assert(await page.locator('.result-number').innerText()!=='0.0200000','100次边界进入融合');
 await page.locator('#walk-reset').click();assert(await page.locator('#walk-exposure').inputValue()==='80','重置样本');
 await page.evaluate(()=>setView('all'));await page.waitForTimeout(200);
 assert(await page.locator('.all-section').count()===35,'连续阅读包含35图');
 assert(await page.locator('#main-view .graph-node').count()===data.meta.node_count,'连续阅读覆盖204节点');
 const sumEdges=data.sections.reduce((a,s)=>a+s.edges.length,0);assert(await page.locator('#main-view .graph-path').count()===sumEdges,'连续阅读保留全部真实连线');
 await overflow(page,'连续阅读');
 await page.evaluate(()=>setView('module','three_trees'));await page.waitForTimeout(80);await page.screenshot({path:path.join(root,'qa','desktop-three-inputs.png'),fullPage:true});
 await page.reload();assert((await page.locator('#main-view h1').innerText()).includes('三种冻结表示树'),'链接刷新恢复模块');
 await page.setViewportSize({width:390,height:844});
 await page.evaluate(()=>setView('overview'));await page.screenshot({path:path.join(root,'qa','mobile-overview.png')});
 await overflow(page,'手机总览');
 await page.locator('#menu-toggle').click();assert(await page.locator('#sidebar').evaluate(n=>n.classList.contains('open')),'手机导航打开');
 await page.locator('#global-search').fill('1017');assert(await page.locator('#sidebar').evaluate(n=>n.classList.contains('open')),'手机搜索期间保持导航打开');
 await page.locator('#nav .nav-item').first().click();assert(!await page.locator('#sidebar').evaluate(n=>n.classList.contains('open')),'手机选中模块后关闭导航');
 for(const s of data.sections){await page.evaluate(id=>setView('module',id),s.id);await overflow(page,'手机 '+s.id);}
 await page.evaluate(()=>setView('module','parallel_day'));await page.waitForTimeout(100);await page.screenshot({path:path.join(root,'qa','mobile-parallel.png'),fullPage:true});
 await page.locator('.graph-node').first().click();assert(await page.locator('#node-dialog').isVisible(),'手机节点可点开');await page.keyboard.press('Escape');
 await page.evaluate(()=>{preparePrint()});await page.emulateMedia({media:'print'});await page.evaluate(()=>drawGraphs());await page.waitForTimeout(100);
 assert(await page.locator('#print-content .print-section').count()===38,'打印包含35图和3张表');
 const printEdges=await page.locator('#print-content .graph-path').count();assert(printEdges===sumEdges,'打印保留真实连线，无虚构顺序箭头',{printEdges,sumEdges});
 await page.pdf({path:path.join(root,'qa','print-preview.pdf'),format:'A4',printBackground:true});
 assert(requests.every(x=>x.startsWith('file:')||x.startsWith('data:')),'自包含，无外部资源请求',requests);
 assert(errors.length===0,'无浏览器脚本或控制台错误',errors);
 await browser.close();
 fs.writeFileSync(path.join(root,'qa','validation.json'),JSON.stringify({checkedAt:new Date().toISOString(),checks,errors,summary:{passed:checks.length,modules:35,nodes:data.meta.node_count,edges:sumEdges}},null,2));
 console.log(JSON.stringify({passed:checks.length,modules:35,nodes:data.meta.node_count,edges:sumEdges,errors}));
})().catch(e=>{fs.writeFileSync(path.join(root,'qa','validation-failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2));console.error(e);process.exitCode=1;});
