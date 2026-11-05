const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname,m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8')),source=JSON.parse(fs.readFileSync(path.join(root,'..','integrated_model.netron'),'utf8')).graphs[0];
const checks=[],errors=[];const clean=s=>s.replace(/\s/g,'');function check(ok,title,detail){checks.push({title,pass:!!ok,detail});if(!ok)throw Error(title+': '+JSON.stringify(detail));}
let browser;
(async()=>{
 browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1600,height:1100}});
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('file:///'+root.replace(/\\/g,'/')+'/index.html');await page.waitForFunction(()=>window.FLOWCHART_VIEWER);
 check(await page.evaluate(()=>FLOWCHART_VIEWER.selected==='paper'),'默认打开论文主图');
 const views=[...m.figures,...m.stages,...m.details];
 for(const view of views){
  await page.evaluate(id=>FLOWCHART_VIEWER.choose(id),view.id);await page.waitForTimeout(20);await page.evaluate(()=>FLOWCHART_VIEWER.zoom(1));
  const metrics=await page.evaluate(()=>{
   const svg=document.querySelector('#surface svg');
   const manual=[...svg.querySelectorAll('.paper-block,.tensor')];
   const groups=manual.length?manual:[...svg.querySelectorAll('g.node')];
   const bad=groups.flatMap(g=>{
    let box;if(g.dataset.w)box={x:+g.dataset.x,y:+g.dataset.y,r:+g.dataset.x+(+g.dataset.w),b:+g.dataset.y+(+g.dataset.h)};
    else{const shapes=[...g.querySelectorAll('polygon,path')].map(p=>p.getBBox());if(!shapes.length)return [];box=shapes.reduce((a,b)=>({x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),r:Math.max(a.r,b.x+b.width),b:Math.max(a.b,b.y+b.height)}),{x:Infinity,y:Infinity,r:-Infinity,b:-Infinity});}
    return [...g.querySelectorAll('text')].flatMap(t=>{const b=t.getBBox();return b.x<box.x-2||b.x+b.width>box.r+2||b.y<box.y-2||b.y+b.height>box.b+2?[{id:g.id,text:t.textContent,box,b:{x:b.x,y:b.y,width:b.width,height:b.height}}]:[];});
   });
   const text=[...svg.querySelectorAll('text')].map(t=>t.textContent).join(' ');
   const outside=[...svg.querySelectorAll('text')].filter(t=>{const b=t.getBoundingClientRect(),s=svg.getBoundingClientRect();return b.left<s.left-2||b.right>s.right+2||b.top<s.top-2||b.bottom>s.bottom+2;}).map(t=>t.textContent);
   return {bad,text,outside,blocks:groups.length};
  });
  check(metrics.bad.length===0,'文字均在模块框内 '+view.id,metrics.bad.slice(0,12));
  check(metrics.outside.length===0,'文字不超出画布 '+view.id,metrics.outside);
  if(view.id==='all'){
   check(await page.locator('#surface g.node').count()===93,'完整图4输入、88计算节点、1输出');
   check(await page.locator('#surface g.edge').count()===146,'完整图146条真实连接');
   const nodes=await page.evaluate(()=>[...document.querySelectorAll('#surface g.node')].map(g=>({id:g.id,text:[...g.querySelectorAll('text')].map(t=>t.textContent).join(' ')})));
   source.nodes.forEach((n,i)=>{const t=clean(nodes.find(x=>x.id==='node-'+i).text);check(t.includes(clean(n.name))&&n.attributes.every(a=>t.includes(clean(a.name+'：'+a.value))),'完整名称和所有参数 '+(i+1));});
   const prod=new Map();source.inputs.forEach((a,i)=>a.value.forEach(v=>prod.set(v,'port_i'+i)));source.nodes.forEach((n,i)=>n.outputs.forEach(a=>a.value.forEach(v=>prod.set(v,'n'+i))));
   const expected=source.nodes.flatMap((n,i)=>n.inputs.flatMap(a=>a.value.map(v=>prod.get(v)+'->n'+i))).concat('n87->out');const actual=new Set(await page.locator('#surface g.edge title').allTextContents());
   check(expected.every(e=>actual.has(e))&&actual.size===new Set(expected).size,'每条真实依赖与原模型一致');
  }
 }
 for(const id of ['paper','neural','final']){
  await page.evaluate(id=>FLOWCHART_VIEWER.choose(id),id);await page.waitForTimeout(60);await page.screenshot({path:path.join(root,'qa',id+'-preview.png')});
 }
 await page.evaluate(()=>FLOWCHART_VIEWER.choose('paper'));await page.waitForTimeout(40);await page.locator('[data-view="detail:twelve"]').click();check(await page.evaluate(()=>FLOWCHART_VIEWER.selected==='detail:twelve'),'点击论文主图分组进入内部结构');
 await page.evaluate(()=>FLOWCHART_VIEWER.choose('all'));await page.waitForTimeout(40);await page.locator('#node-86').click();check(await page.locator('#inspector').isVisible()&&(await page.locator('#node-attrs').innerText()).includes('401'),'完整图点击模型显示原始属性');await page.locator('#open-detail').click();check(await page.evaluate(()=>FLOWCHART_VIEWER.selected.startsWith('detail:')),'完整图能展开模块');
 const scale=await page.evaluate(()=>FLOWCHART_VIEWER.scale);await page.locator('#zoom-in').click();check(await page.evaluate(()=>FLOWCHART_VIEWER.scale)>scale,'缩放按钮工作');
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>FLOWCHART_VIEWER.choose('paper'));await page.waitForTimeout(70);check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'手机页面无横向溢出');await page.screenshot({path:path.join(root,'qa','mobile-preview.png')});
 check(errors.length===0,'无浏览器脚本错误',errors);
 for(const item of views){check(fs.statSync(path.join(root,item.file)).size>1000,'SVG存在 '+item.id);check(fs.statSync(path.join(root,item.file.replace(/\.svg$/,'.png'))).size>1000,'PNG存在 '+item.id);}
 fs.writeFileSync(path.join(root,'qa','validation.json'),JSON.stringify({at:new Date().toISOString(),passed:checks.length,checks,errors},null,2));console.log(JSON.stringify({passed:checks.length,graphs:views.length,nodes:88,edges:146,errors}));await browser.close();
})().catch(async e=>{fs.writeFileSync(path.join(root,'qa','validation-failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2));console.error(e);await browser?.close();process.exitCode=1;});
