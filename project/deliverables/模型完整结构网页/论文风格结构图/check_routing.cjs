const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname;
(async()=>{
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1200}}),reports=[];
  await page.goto('file:///'+root.replace(/\\/g,'/')+'/index.html');
  await page.waitForFunction(()=>window.FLOWCHART_VIEWER);
  for(const id of ['paper','neural','final']){
   await page.evaluate(id=>FLOWCHART_VIEWER.choose(id),id);
   const report=await page.evaluate(()=>{
    const svg=document.querySelector('#surface svg'),paths=[...svg.querySelectorAll('path.connection')];
    const rect=b=>({x:b.x,y:b.y,w:b.width,h:b.height});
    const nodes=[...svg.querySelectorAll('.paper-block,.tensor,.operator')].map(g=>({name:g.textContent,b:{x:+g.dataset.x,y:+g.dataset.y,w:+g.dataset.w,h:+g.dataset.h},shapes:[...g.querySelectorAll('rect,polygon,circle')]}));
    const text=[...svg.querySelectorAll('text')].map(t=>({name:t.textContent,b:rect(t.getBBox())}));
    const inBox=(p,b,m=0)=>p.x>b.x+m&&p.x<b.x+b.w-m&&p.y>b.y+m&&p.y<b.y+b.h-m;
    const hitNodes=[],hitText=[],diagonal=[],looseEnds=[],outside=[],crossings=[],crowdedArrowheads=[],segments=[],ends=[];
    paths.forEach((p,i)=>{
     const pts=JSON.parse(p.dataset.points),seenN=new Set(),seenT=new Set(),len=p.getTotalLength();
     pts.slice(1).forEach((v,j)=>segments.push({i,a:pts[j],b:v}));
     pts.slice(1).forEach((v,j)=>{if(v[0]!==pts[j][0]&&v[1]!==pts[j][1])diagonal.push({i,points:[pts[j],v]});});
     for(let d=3;d<len-3;d+=2){
      const point=p.getPointAtLength(d);
      if(point.x<0||point.y<0||point.x>svg.viewBox.baseVal.width||point.y>svg.viewBox.baseVal.height)outside.push(i);
      for(const n of nodes)if(!seenN.has(n)&&inBox(point,n.b,2.5)&&n.shapes.some(s=>s.isPointInFill(point))){seenN.add(n);hitNodes.push({i,node:n.name,point:rect({x:point.x,y:point.y,width:0,height:0}),points:pts});}
      for(const t of text)if(!seenT.has(t)&&inBox(point,t.b,-1.5)){seenT.add(t);hitText.push({i,text:t.name,at:[point.x,point.y],points:pts});}
     }
     if(p.hasAttribute('marker-end')){
      ends.push({i,p:pts[pts.length-1]});
      const point=p.getPointAtLength(len),near=nodes.some(n=>point.x>=n.b.x-3&&point.x<=n.b.x+n.b.w+3&&point.y>=n.b.y-3&&point.y<=n.b.y+n.b.h+3&&!(point.x>n.b.x+4&&point.x<n.b.x+n.b.w-4&&point.y>n.b.y+4&&point.y<n.b.y+n.b.h-4));
      if(!near)looseEnds.push({i,end:[point.x,point.y],points:pts});
     }
    });
    const dots=[...svg.querySelectorAll('.junction')].map(n=>[+n.getAttribute('cx'),+n.getAttribute('cy')]);
    for(let i=0;i<segments.length;i++)for(let j=i+1;j<segments.length;j++){
     const a=segments[i],b=segments[j];if(a.i===b.i)continue;
     const ah=a.a[1]===a.b[1],bh=b.a[1]===b.b[1];if(ah===bh)continue;
     const h=ah?a:b,v=ah?b:a,x=v.a[0],y=h.a[1];
     if(x>Math.min(h.a[0],h.b[0])+.1&&x<Math.max(h.a[0],h.b[0])-.1&&y>Math.min(v.a[1],v.b[1])+.1&&y<Math.max(v.a[1],v.b[1])-.1&&!dots.some(p=>Math.hypot(p[0]-x,p[1]-y)<3))crossings.push({lines:[a.i,b.i],point:[x,y]});
    }
    for(let i=0;i<ends.length;i++)for(let j=i+1;j<ends.length;j++){
     const a=ends[i],b=ends[j],distance=Math.hypot(a.p[0]-b.p[0],a.p[1]-b.p[1]);
     if(distance<18)crowdedArrowheads.push({lines:[a.i,b.i],distance});
    }
    return {wires:paths.length,junctions:svg.querySelectorAll('.junction').length,hitNodes,hitText,diagonal,looseEnds,outside,crossings,crowdedArrowheads};
   });
   reports.push({id,...report});
  }
  const problems=reports.reduce((s,r)=>s+r.hitNodes.length+r.hitText.length+r.diagonal.length+r.looseEnds.length+r.outside.length+r.crowdedArrowheads.length+(['paper','final'].includes(r.id)?r.crossings.length:0),0);
  fs.writeFileSync(path.join(root,'qa','routing-validation.json'),JSON.stringify({at:new Date().toISOString(),problems,reports},null,2));
  console.log(JSON.stringify({problems,reports}));
  if(problems)process.exitCode=1;
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
