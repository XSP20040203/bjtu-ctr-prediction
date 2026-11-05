(() => {
  'use strict';
  const {manifest, svgs, nodes}=window.FLOWCHART_DATA;
  const $=id=>document.getElementById(id);
  const canvas=$('canvas'),surface=$('surface');
  let selected='full',scale=1,width=0,height=0,drag=null,dragged=false,shownNode=null;
  const views=[{id:'full',title:'完整总流程图',...manifest.full},...manifest.stages,...manifest.details.map(s=>({...s,id:'detail:'+s.id}))];
  const getView=()=>views.find(v=>v.id===selected)||views[0];
  manifest.stages.forEach((s,i)=>{
    const b=document.createElement('button');b.className='nav-item';b.dataset.view=s.id;b.textContent=String(i+1).padStart(2,'0')+'  '+s.title;$('stage-nav').append(b);
  });
  manifest.details.forEach(s=>{const o=document.createElement('option');o.value='detail:'+s.id;o.textContent=s.title;$('details').append(o);});
  for(const v of views){const o=document.createElement('option');o.value=v.id;o.textContent=v.title;$('mobile-view').append(o);}
  function syncZoom(){
    const svg=surface.querySelector('svg');if(!svg)return;
    svg.setAttribute('width',width);svg.setAttribute('height',height);
    svg.style.transform=`scale(${scale})`;
    surface.style.width=(width*scale)+'px';surface.style.height=(height*scale)+'px';
    $('zoom-label').textContent=Math.round(scale*100)+'%';
  }
  function zoom(value,x=canvas.clientWidth/2,y=canvas.clientHeight/2){
    const ox=(canvas.scrollLeft+x-24)/scale,oy=(canvas.scrollTop+y-24)/scale;
    scale=Math.max(.025,Math.min(3,value));syncZoom();
    canvas.scrollLeft=ox*scale-x+24;canvas.scrollTop=oy*scale-y+24;
  }
  function fit(){scale=Math.min(1.05,(canvas.clientWidth-56)/width,(canvas.clientHeight-56)/height);syncZoom();canvas.scrollLeft=0;canvas.scrollTop=0;}
  function choose(id,updateHash=true){
    if(!svgs[id])id='full';selected=id;const v=getView();width=v.width;height=v.height;
    surface.innerHTML=svgs[id];$('inspector').hidden=true;shownNode=null;
    document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-current',b.dataset.view===selected?'true':'false'));
    $('details').value=selected.startsWith('detail:')?selected:'';$('mobile-view').value=selected;
    $('download-svg').href=v.file;$('download-svg').download=v.title+'.svg';
    $('download-png').href=v.file.replace(/\.svg$/,'.png');$('download-png').download=v.title+'.png';
    const index=manifest.stages.findIndex(s=>s.id===selected);
    $('previous').disabled=index===0;$('next').disabled=index===manifest.stages.length-1;
    const line=selected==='full'?'四张输入表 → 88 个计算模块 → id、pctr 提交文件':(v.note||'模块内部的完整流程、维度和连接关系');
    $('status').textContent=v.title+' ｜ '+line;
    if(updateHash)history.replaceState(null,'','#'+encodeURIComponent(id));
    requestAnimationFrame(fit);
  }
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>choose(b.dataset.view)));
  $('details').addEventListener('change',e=>{if(e.target.value)choose(e.target.value);});
  $('mobile-view').addEventListener('change',e=>choose(e.target.value));
  $('fit').onclick=fit;$('actual').onclick=()=>zoom(1);$('zoom-in').onclick=()=>zoom(scale*1.4);$('zoom-out').onclick=()=>zoom(scale/1.4);
  function adjacent(step){let i=manifest.stages.findIndex(s=>s.id===selected);if(i===-1)i=step>0?-1:manifest.stages.length;choose(manifest.stages[Math.max(0,Math.min(manifest.stages.length-1,i+step))].id);}
  $('previous').onclick=()=>adjacent(-1);$('next').onclick=()=>adjacent(1);
  $('dismiss-note').onclick=()=>{$('read-note').hidden=true;};
  function inspect(id){
    const n=nodes[id];if(!n)return;shownNode=n;
    surface.querySelectorAll('.picked').forEach(g=>g.classList.remove('picked'));surface.querySelector('#node-'+id)?.classList.add('picked');
    $('node-title').textContent=String(id+1).padStart(2,'0')+'  '+n.name;$('node-attrs').replaceChildren();
    n.attributes.forEach(a=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=a.name;dd.textContent=a.value;$('node-attrs').append(dt,dd);});
    $('node-description').textContent=n.description;$('open-detail').hidden=!n.detail;$('inspector').hidden=false;
  }
  $('open-detail').onclick=()=>{if(shownNode?.detail)choose('detail:'+shownNode.detail);};
  $('close-inspector').onclick=()=>{$('inspector').hidden=true;};
  canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect();zoom(scale*Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={id:e.pointerId,x:e.clientX,y:e.clientY,left:canvas.scrollLeft,top:canvas.scrollTop};dragged=false;canvas.classList.add('dragging');});
  window.addEventListener('pointermove',e=>{if(!drag||e.pointerId!==drag.id)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)dragged=true;canvas.scrollLeft=drag.left-dx;canvas.scrollTop=drag.top-dy;});
  window.addEventListener('pointerup',()=>{drag=null;canvas.classList.remove('dragging');});
  window.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('dragging');});
  canvas.addEventListener('click',e=>{
    if(dragged)return;const g=e.target.closest('g.node');if(!g)return;
    const id=g.id.match(/^node-(\d+)$/);if(id)inspect(Number(id[1]));
    const from=g.id.match(/^from-(\d+)$/),to=g.id.match(/^to-(\d+)$/);
    if(from){choose(manifest.stages[nodes[Number(from[1])].stage].id);inspect(Number(from[1]));}
    if(to){const d=Number(to[1]);const next=manifest.stages.find(s=>s.id!==selected&&svgs[s.id].includes('id="from-'+d+'"'));if(next)choose(next.id);}
  });
  canvas.addEventListener('dblclick',e=>{const r=canvas.getBoundingClientRect();zoom(scale*1.5,e.clientX-r.left,e.clientY-r.top);});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')$('inspector').hidden=true;});
  window.addEventListener('hashchange',()=>choose(decodeURIComponent(location.hash.slice(1)),false));
  window.FLOWCHART_VIEWER={choose,fit,zoom,get selected(){return selected;},get scale(){return scale;},getView,inspect};
  choose(decodeURIComponent(location.hash.slice(1))||'full',false);
})();
