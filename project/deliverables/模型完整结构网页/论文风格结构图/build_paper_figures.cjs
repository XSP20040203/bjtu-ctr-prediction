/* Original paper-style drawings. Reads documented architecture; never runs ML code. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {instance}=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@viz-js/viz');
const ROOT=__dirname,SOURCE=path.dirname(ROOT);
const raw=fs.readFileSync(path.join(SOURCE,'integrated_model.netron'),'utf8');
const model=JSON.parse(raw),main=model.graphs[0];
const guide=JSON.parse(fs.readFileSync(path.join(SOURCE,'model_data.json'),'utf8'));
const previous=JSON.parse(fs.readFileSync(path.join(SOURCE,'完整流程图','manifest.json'),'utf8'));
const stages=previous.stages.map(({id,title,ids,note})=>({id,title,ids,note}));
const P={data:['#dcecf7','#547b94'],net:['#dfdef3','#716a9d'],tree:['#dff0e1','#66866b'],merge:['#ffedcc','#b69a65'],gate:['#f6dfe2','#ae7e88'],muted:['#f3f4f5','#9aabb6']};
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const q=s=>JSON.stringify(String(s));
function wrap(s,max){const out=[];for(const line of String(s).split('\n')){let t='',u=0;for(const token of line.match(/[A-Za-z0-9_]+(?:\.[0-9]+)*|[^A-Za-z0-9_]/gu)||['']){const n=[...token].reduce((a,c)=>a+(c.codePointAt(0)>255?1:.53),0);if(t&&u+n>max){out.push(t.trim());t='';u=0;}t+=token;u+=n;}out.push(t.trim());}return out;}
class Fig{
 constructor(w,h,title,subtitle){this.w=w;this.h=h;this.title=title;this.back=[];this.wires=[];this.junctions=[];this.front=[];this.text(40,52,title,34,'start',600);this.text(42,90,subtitle,18,'start',400,'#607080');}
 text(x,y,s,size=19,anchor='middle',weight=400,color='#263746',dest=this.front){dest.push(`<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${color}">${esc(s)}</text>`);}
 lines(x,y,lines,size=18,anchor='middle',weight=400,color='#263746',dest=this.front){lines.forEach((s,i)=>this.text(x,y+i*size*1.35,s,size,anchor,weight,color,dest));}
 panel(x,y,w,h,title,kind='muted'){const [,stroke]=P[kind];this.back.push(`<rect class="panel" x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${stroke}" stroke-width="1.3" stroke-dasharray="8 5"/>`);this.text(x+20,y+33,title,23,'start',600,stroke);}
 box(x,y,w,h,title,body='',kind='net',extra={}){
  const [fill,stroke]=P[kind],ts=extra.ts||18,bs=extra.bs||16;
  const titles=wrap(title,(w-24)/ts),bodies=body?wrap(body,(w-24)/bs):[];
  const linesHeight=titles.length*ts*1.32+(bodies.length?10:0)+bodies.length*bs*1.33;
  const start=y+(h-linesHeight)/2+ts;
  const group=[];
  group.push(`<g class="paper-block" ${extra.id?'id="'+esc(extra.id)+'"':''} ${extra.link?'data-view="'+esc(extra.link)+'"':''} data-x="${x}" data-y="${y}" data-w="${w}" data-h="${h}"><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="#6d7680" stroke-width="1.1"/>`);
  this.lines(x+w/2,start,titles,ts,'middle',500,'#223746',group);
  if(bodies.length)this.lines(x+w/2,start+titles.length*ts*1.32+10,bodies,bs,'middle',400,'#425968',group);
  group.push('</g>');this.front.push(group.join(''));return {x,y,w,h,cx:x+w/2,cy:y+h/2};
 }
 cube(x,y,w,h,title,dimension,kind='data',extra={}){
  const [fill,stroke]=P[kind],d=12;
  this.front.push(`<g class="tensor" data-x="${x}" data-y="${y-d}" data-w="${w+d}" data-h="${h+d}"><rect x="${x}" y="${y-d}" width="${w+d}" height="${h+d}" fill="${fill}" stroke="#6d7680" stroke-width="1.1"/>`);
  this.lines(x+w/2,y+h*.38,wrap(title,(w-18)/(extra.ts||18)),extra.ts||18);
  this.text(x+w/2,y+h-15,dimension,extra.ds||19,'middle',500);this.front.push('</g>');return {x,y,w,h,cx:x+w/2,cy:y+h/2};
 }
 op(x,y,label='C',r=21,kind='merge'){const [fill,stroke]=P[kind];this.front.push(`<g class="operator" data-x="${x-r}" data-y="${y-r}" data-w="${r*2}" data-h="${r*2}"><circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/><text x="${x}" y="${y+7}" text-anchor="middle" font-size="23" fill="#384954">${esc(label)}</text></g>`);}
 arrow(points,opt={}){
  const color=opt.color||'#536471';
  if(typeof points==='string')throw Error('Use explicit orthogonal points for auditable routing');
  const pts=points.filter((p,i)=>!i||p[0]!==points[i-1][0]||p[1]!==points[i-1][1]);
  const d='M'+pts.map(p=>p.join(',')).join(' L');
  const attrs=`d="${d}" fill="none" stroke-linecap="round" stroke-linejoin="round"`;
  // White separation makes a crossing without a junction visibly disconnected.
  this.wires.push(`<g class="wire"><path class="wire-clearance" ${attrs} stroke="white" stroke-width="4.5"/><path class="connection" data-points="${esc(JSON.stringify(pts))}" ${attrs} stroke="${color}" stroke-width="${opt.width||1.55}" ${opt.dash?'stroke-dasharray="7 5"':''} ${opt.end===false?'':'marker-end="url(#'+(opt.color?'purple':'arrow')+')"'}/></g>`);
 }
 dot(x,y,color='#536471'){this.junctions.push(`<circle class="junction" cx="${x}" cy="${y}" r="2.4" fill="${color}"/>`);}
 label(x,y,t,size=16,color='#546a78'){this.text(x,y,t,size,'middle',400,color);}
 finish(){return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(this.title)}" width="${this.w}" height="${this.h}" viewBox="0 0 ${this.w} ${this.h}"><defs>${[['arrow','#536471'],['purple','#8175a4']].map(([id,color])=>`<marker id="${id}" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="8" viewBox="0 0 9 8" refX="8" refY="4" orient="auto"><path d="M1,1 L8,4 L1,7 Z" fill="${color}"/></marker>`).join('')}</defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, SimHei, sans-serif"><g class="backgrounds">${this.back.join('\n')}</g><g class="connections">${this.wires.join('\n')}</g><g class="junctions">${this.junctions.join('\n')}</g><g class="foreground">${this.front.join('\n')}</g></g></svg>`;}
}
function overview(){return require('./paper_yolo11_layout.cjs').overview(Fig);}
function neural(){
 const f=new Fig(2840,2015,'核心神经网络模块 · 张量与算子展开','按论文子图展示嵌入、交叉、字段门控、日状态、并行表格网络和用户日总强度分配；B 为广告批次，G 为用户日数');
 f.panel(40,130,1360,510,'(a) 类别与无序历史的嵌入及池化','data');
 f.cube(85,245,200,86,'27 个单值类别','B × 27','data');f.cube(85,430,200,86,'12 个历史列表','B × 12 × 5','data');
 f.box(355,255,235,65,'查询 8 维嵌入表','','net',{ts:21});f.box(355,435,235,65,'查询同类型字典','','net',{ts:21});
 f.cube(670,250,235,83,'单值字段向量','B × 27 × 8','net');f.cube(670,430,235,83,'列表项向量','B × 12 × 5 × 8','net');
 f.box(995,428,280,87,'掩码均值池化','仅平均有效项 → B × 12 × 8','data',{ts:21,bs:18});
 f.op(1135,305,'C',25);
 f.arrow([[297,288],[355,288]]);f.arrow([[590,288],[670,288]]);f.arrow([[297,472],[355,472]]);f.arrow([[590,472],[670,472]]);f.arrow([[917,472],[995,472]]);
 f.arrow([[917,290],[1050,290],[1050,305],[1110,305]]);f.arrow([[1135,428],[1135,330]]);f.arrow([[1160,305],[1335,305],[1335,588],[1052,588]]);
 f.cube(780,555,260,66,'39 个字段拼接','B × 312','net');
 f.label(405,590,'PAD = 0；UNK = 1；同类型候选与历史共用字典',18);
 f.panel(1440,130,1360,510,'(b) 有界编号深度交叉网络：交叉分支 ∥ 深层分支','net');
 f.cube(1485,305,170,100,'拼接输入 x₀','B × D','net');f.label(1570,455,'D = 478 / 637',20);
 f.box(1785,220,450,100,'两层全矩阵交叉','Wₗ 为 D × D；每层输出 B × D','net',{ts:23,bs:21});
 f.box(1785,425,450,90,'深层全连接分支','D → 128 → 64','net',{ts:23,bs:23});
 f.arrow([[1667,355],[1725,355]],{end:false});f.arrow([[1725,355],[1725,270],[1785,270]]);f.arrow([[1725,355],[1725,470],[1785,470]]);f.dot(1725,355);
 f.op(2335,355,'C',26);f.arrow([[2235,270],[2335,270],[2335,329]]);f.arrow([[2235,470],[2335,470],[2335,381]]);
 f.box(2430,300,300,112,'线性输出与 Sigmoid','D + 64 → 1\n542 / 701 → B × 1','merge',{ts:22,bs:21});f.arrow([[2361,355],[2430,355]]);
 f.label(2110,576,'每层：xₗ₊₁ = xₗ + x₀ ⊙ (Wₗ xₗ + bₗ)',23);f.label(2110,610,'基础模型与折外曝光模型 D = 478；多时间尺度模型 D = 637',18);
 f.panel(40,690,1360,650,'(c) 丰富历史字段门控与低秩多专家交叉网络','net');
 f.cube(85,810,210,85,'字段嵌入','B × 39 × 8','net');f.box(380,790,455,110,'字段门控控制器','39 个字段均值 + 数值投影 16 = 55\n55 → 16 → 39 → 2 × Sigmoid','net',{ts:22,bs:20});
 f.op(925,850,'⊙',25);f.arrow([[307,850],[340,850]],{end:false});f.arrow([[340,850],[340,765],[925,765],[925,825]]);f.arrow([[340,850],[380,850]]);f.dot(340,850);f.arrow([[835,850],[900,850]]);
 f.op(1110,850,'C',25);f.arrow([[950,850],[1085,850]]);f.cube(1180,810,155,90,'主输入 x₀','B × 812','net',{ts:19});f.arrow([[1135,850],[1180,850]]);
 f.label(925,930,'每字段增益广播到 8 维',17);
 f.box(85,950,210,78,'500 维数值','清理并裁剪到 ±8','data',{ts:21,bs:18});f.arrow([[295,990],[340,990]],{end:false});f.arrow([[340,990],[1110,990],[1110,875]]);f.arrow([[340,990],[340,940],[605,940],[605,900]]);f.dot(340,990);
 f.label(615,967,'其中 500 → 16 供门控器使用',18);
 f.box(85,1150,170,90,'x₀','D = 812','net',{ts:24,bs:22});
 f.arrow([[1257,900],[1365,900],[1365,1040],[170,1040],[170,1150]]);
 f.panel(325,1060,710,180,'每层 3 个专家，交叉层重复 2 次','net');
 for(let i=0;i<3;i++)f.box(345+i*230,1135,210,70,'低秩专家 '+(i+1),'D → 32 → 32 → D','net',{ts:18,bs:16});
 f.box(345,1250,690,65,'深层分支：812 → 128 → 64；Dropout 0.1','','net',{ts:22});
 f.arrow([[255,1195],[290,1195]],{end:false});f.arrow([[290,1195],[290,1117],[910,1117]],{end:false});f.dot(290,1195);f.arrow([[290,1195],[290,1280],[345,1280]]);
 for(let i=0;i<3;i++){const x=450+i*230;f.arrow([[x,1117],[x,1135]]);f.dot(x,1117);f.arrow([[x,1205],[x,1222]],{end:false});f.dot(x,1222);}
 f.arrow([[450,1222],[1025,1222],[1025,1140],[1045,1140]]);
 f.box(1045,1060,160,130,'专家加权与交叉残差','x + x₀ ⊙ mixed','net',{ts:18,bs:16});
 f.op(1130,1240,'C',24);f.arrow([[1125,1190],[1125,1205],[1130,1205],[1130,1216]]);f.arrow([[1035,1280],[1130,1280],[1130,1264]]);
 f.box(1190,1190,160,105,'线性与 Sigmoid','876 → 1\n输出 B × 1','merge',{ts:18,bs:18});f.arrow([[1154,1240],[1190,1240]]);
 f.label(680,1331,'每专家两次 tanh；路由 D → 3 → Softmax；mixed = Σₑ aₑ expertₑ(x)',15);
 f.panel(1440,690,1360,650,'(d) 用户日状态保护网络：广告分数 + 有界日状态增量','net');
 f.cube(1485,815,210,85,'广告特征','B × 478','net');f.box(1770,800,480,110,'门控低秩交叉与深层主干','478 → 拼接 542 → 1 原始分数','net',{ts:23,bs:21});f.arrow([[1707,855],[1770,855]]);
 f.op(2525,1000,'+',26);f.arrow([[2250,855],[2525,855],[2525,974]]);f.box(2590,950,170,100,'Sigmoid','B × 1','merge',{ts:21,bs:22});f.arrow([[2551,1000],[2590,1000]]);
 f.cube(1485,1065,210,80,'无标签日上下文','G × 598','data');f.box(1485,1190,210,70,'用户嵌入','G × 8','net',{ts:21,bs:21});
 f.op(1765,1110,'C',24);f.arrow([[1707,1105],[1725,1105],[1725,1035],[1765,1035],[1765,1086]]);f.arrow([[1707,1225],[1765,1225],[1765,1134]]);
 f.box(1830,1060,345,100,'日状态编码器','606 → 64 → 32\nG × 32','net',{ts:22,bs:22});f.arrow([[1789,1110],[1830,1110]]);
 f.box(2235,1075,200,73,'按组索引回取','B × 32','data',{ts:20,bs:21});f.arrow([[2175,1110],[2235,1110]]);
 f.box(2235,1200,260,84,'行状态修正头','32 → 1 → 2 × tanh','net',{ts:22,bs:20});f.arrow([[2335,1148],[2335,1200]]);f.arrow([[2495,1242],[2525,1242],[2525,1026]]);
 f.box(1830,1210,345,74,'仅训练的日点击辅助头','32 → 1；BCE 权重 0.02','muted',{ts:21,bs:19});f.arrow([[2002,1160],[2002,1210]],{dash:true});
 f.label(2100,1313,'保护参考、正样本加权、成对排序微调：三份检查点，各自保存独立权重',19);
 f.panel(40,1390,1360,540,'(e) 分段数值嵌入并行表格神经网络 TabM','net');
 f.cube(85,1520,190,90,'数值输入','B × 535','data');f.box(330,1515,280,105,'逐数值分段线性嵌入','535 × 4 = 2140\n输出 B × 2140','net',{ts:21,bs:21});f.arrow([[287,1565],[330,1565]]);
 f.cube(85,1730,190,85,'类别列表嵌入','B × 312','net');f.op(695,1670,'C',25);f.arrow([[610,1567],[695,1567],[695,1645]]);f.arrow([[287,1772],[695,1772],[695,1695]]);
 f.cube(765,1615,200,110,'拼接并复制','B × 4 × 2452','net',{ts:20,ds:20});f.arrow([[720,1670],[765,1670]]);
 f.arrow([[977,1670],[1000,1670]],{end:false});f.arrow([[1000,1514],[1000,1784]],{end:false});f.dot(1000,1670);
 f.arrow([[1360,1514],[1360,1860],[1305,1860]]);
 for(let i=0;i<4;i++){const y=1480+i*90;f.box(1040,y,290,68,'成员 '+(i+1)+'：2452 → 128 → 128','1 个分数 → Sigmoid','net',{ts:18,bs:17});f.arrow([[1000,y+34],[1040,y+34]]);f.dot(1000,y+34);f.arrow([[1330,y+34],[1360,y+34]],{end:false});f.dot(1360,y+34);}
 f.op(1280,1860,'μ',25);f.label(1120,1910,'4 个概率平均 → B × 1；两层 Dropout 0.15',19);
 f.panel(1440,1390,1360,540,'(f) 用户日总强度分配网络：先估计总量，再分给每条广告','net');
 f.box(1485,1490,255,100,'输入拼接','312 嵌入 + 500→32 数值\n344 → 128 → 64','net',{ts:22,bs:19});
 f.cube(1810,1500,185,85,'广告 token','B × 64','net');f.arrow([[1740,1540],[1810,1540]]);
 f.box(2070,1470,290,100,'4 个学习槽注意力汇总','槽宽 32；键 64 → 32\nG × 4 × 64','net',{ts:21,bs:20});f.arrow([[2007,1542],[2035,1542],[2035,1520],[2070,1520]]);
 f.box(2070,1625,290,90,'候选读取槽上下文','查询 64→32；每行 B × 64','net',{ts:21,bs:19});f.arrow([[2215,1570],[2215,1600]],{end:false});f.arrow([[2215,1600],[2215,1625]]);f.dot(2215,1600);
 f.box(2440,1470,300,105,'行分配分数头','token、均值、上下文、乘积\n256 → 64 → 1','net',{ts:22,bs:20});f.arrow([[2360,1670],[2390,1670],[2390,1522],[2440,1522]]);
 f.box(1810,1765,550,95,'用户日摘要与总量头','均值64 + 槽均值64 + 最大64 + log组大小1 = 193\n偏置头与计数头：193 → 32 → 1 → λ','net',{ts:22,bs:19});f.arrow([[1902,1585],[1902,1765]]);f.arrow([[2215,1600],[2025,1600],[2025,1765]]);
 f.box(2440,1655,300,90,'组内 Softmax 分配','份额 × 用户日总强度 λ','merge',{ts:22,bs:21});f.arrow([[2590,1575],[2590,1655]]);f.arrow([[2360,1812],[2400,1812],[2400,1700],[2440,1700]]);
 f.box(2440,1815,300,80,'强度转换为点击概率','p = 1 − exp(−行强度)','merge',{ts:21,bs:20});f.arrow([[2590,1745],[2590,1815]]);
 f.text(40,1980,'图 2　核心网络模块展开。C：拼接；⊙：逐元素乘积；+：相加；μ：均值。平面特征块标注真实张量宽度，箭头贴合边框。',19,'start');
 return f;
}
function finalRepresentation(){return require('./paper_yolo11_layout.cjs').finalRepresentation(Fig);}
const producer=new Map();main.inputs.forEach((a,i)=>a.value.forEach(v=>producer.set(v,'i'+i)));main.nodes.forEach((n,i)=>n.outputs.forEach(a=>a.value.forEach(v=>producer.set(v,i))));
const edges=[];main.nodes.forEach((n,i)=>n.inputs.forEach((a,j)=>a.value.forEach(v=>edges.push({from:producer.get(v),to:i,value:v,input:j}))));
const stageOf=new Map(stages.flatMap((s,i)=>s.ids.map(id=>[id,i])));
const dim=v=>(main.values[v].type?.shape?.dimensions||[]).join(' × ');
function compactNode(id){
 const n=main.nodes[id],kind=({Data:'data',Transform:'data',Layer:'net',Tree:'tree',Merge:'merge',Normalization:'merge',Control:'gate'})[n.type.category]||'data';
 const [fill,stroke]=P[kind],rows=[...wrap(n.name,21),...n.attributes.flatMap(a=>wrap(a.name+'：'+a.value,29))];
 let label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" WIDTH="400">';
 const t=wrap(n.name,21);label+='<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="16"><B>'+t.map(esc).join('<BR ALIGN="LEFT"/>')+'</B></FONT></TD></TR>';
 for(const a of n.attributes)label+='<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12">'+wrap(a.name+'：'+a.value,29).map(esc).join('<BR ALIGN="LEFT"/>')+'</FONT></TD></TR>';
 label+='</TABLE>>';
 if(n.type.category==='Control')label=q(rows.join('\n'));
 return `n${id} [id="node-${id}" shape=${n.type.category==='Control'?'diamond':'box'} ${n.type.category==='Control'?'width=8,':''} style="rounded,filled" fillcolor="${fill}" color="${stroke}" penwidth=1.2 margin="0.15,0.12" label=${label}];\n`;
}
function dotHead(title){return `digraph p {graph[rankdir=TB,bgcolor=white,labelloc=t,label=${q(title)},fontname="Microsoft YaHei",fontsize=24,nodesep=.42,ranksep=.65,splines=polyline,outputorder=edgesfirst,newrank=true,pad=.35];node[fontname="Microsoft YaHei",fontsize=14];edge[color="#647480",penwidth=1.15,arrowsize=.65,fontname="Microsoft YaHei",fontsize=12];\n`;}
function ranks(s){const rows=s.id==='foundation'?[[3,4,5,6],[7,8,9,10],[12,13,14,15]]:s.id==='day'?[[26,27,28,29],[30,31,32,33],[34,35,36,37]]:[];let d='';for(const r of rows)d+='{rank=same;'+r.map(i=>'n'+i).join(';')+';}\n';for(let i=1;i<rows.length;i++)for(let c=0;c<rows[i].length;c++)d+=`n${rows[i-1][c]}->n${rows[i][c]}[style=invis,weight=50];\n`;return d;}
function stageDot(s,idx,all=false){
 const ids=new Set(s.ids);let d=dotHead(all?'整合点击率模型 · 全部 88 个计算模块与完整参数':String(idx+1).padStart(2,'0')+'  '+s.title);
 for(const id of s.ids)d+=compactNode(id);
 if(all)for(const s of stages)d+=ranks(s);else d+=ranks(s);
 const external=new Map();for(const e of edges)if(ids.has(e.to)&&!ids.has(e.from))external.set(e.from,e.value);
 for(const [id,v]of external){const isInput=typeof id==='string',title=isInput?main.inputs[Number(id.slice(1))].name:main.nodes[id].name,origin=isInput?'原始输入表':`来自第${stageOf.get(id)+1}部分`;
 d+=`port_${id}[id="port-${id}",shape=box,width=5,style="rounded,filled",fillcolor="#f3f4f5",color="#a8b3bc",label=${q(origin+'\n'+wrap(title,23).join('\n')+'\n'+dim(v))}];\n`;}
 for(const e of edges)if(ids.has(e.to))d+=`${ids.has(e.from)?'n'+e.from:'port_'+e.from}->n${e.to};\n`;
 if(ids.has(87))d+='out[shape=box,width=3,style="rounded,filled",fillcolor="#ffedcc",label="N × 1 点击率\\nid、pctr 提交文件"];n87->out;\n';
 return d+'}';
}
function detailDot(s){let d=dotHead(s.title);s.nodes.forEach((n,i)=>{let content=n.body;if(s.id==='embedding'&&i===2)content+='\n不同网络成员权重独立';d+=`d${i}[id="detail-node-${i}",shape=box,width=6.6,style="rounded,filled",fillcolor="#dfdef3",color="#827aa6",margin="0.18,0.14",label=${q([...wrap(n.title,24),...content.split('\n').flatMap(s=>wrap(s,31))].join('\n'))}];\n`;});for(const row of s.rows)if(row.length>1)d+='{rank=same;'+row.map(i=>'d'+i).join(';')+';}\n';for(const [a,b]of s.edges)d+=`d${a}->d${b};\n`;return d+'}';}
async function build(){
 for(const name of ['sections','details','qa'])fs.mkdirSync(path.join(ROOT,name),{recursive:true});
 const manifest={sourceSha256:crypto.createHash('sha256').update(raw).digest('hex'),nodes:88,edges:146,figures:[],stages:[],details:[],references:[{paper:'YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review',figure:'Figure 4; module details in Figures 15, 16 and 18',url:'https://arxiv.org/pdf/2501.13400v4'},{paper:'You Only Look Once: Unified, Real-Time Object Detection',figure:'Figure 3',url:'https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf'},{paper:'YOLOv7 supplementary material',figure:'Figures A3–A4',url:'https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_YOLOv7_Trainable_Bag-of-Freebies_CVPR_2023_supplemental.pdf'}]};
 const views={};
 function saveManual(id,file,fig,note){const svg=fig.finish();fs.writeFileSync(path.join(ROOT,file),svg);const item={id,title:fig.title,file,width:fig.w,height:fig.h,note};manifest.figures.push(item);views[id]=svg;}
 saveManual('paper','论文式模型整体结构图.svg',overview(),'按论文主图组织：特征块、并行主干、用户日双支路、统计增强、冻结表示与条件输出。');
 saveManual('neural','核心神经网络模块展开图.svg',neural(),'六个子图逐层展示真实维度和张量算子。');
 saveManual('final','冻结表示与最终融合结构图.svg',finalRepresentation(),'三份独立输入与四个末端条件融合模块。');
 const viz=await instance();
 function saveDot(id,file,title,dot){const result=viz.render(dot,{format:'svg'});if(result.status!=='success')throw Error(JSON.stringify(result.errors));const svg=result.output.replace(/<\?xml[^>]*>\s*/,'').replace(/<!DOCTYPE[\s\S]*?>\s*/,'').replace(/font-family="Microsoft YaHei"/g,'font-family="Microsoft YaHei, SimHei, sans-serif"');fs.writeFileSync(path.join(ROOT,file),svg);views[id]=svg;const vb=svg.match(/viewBox="([^"]+)"/)[1].split(' ').map(Number);return {id,file,title,width:vb[2],height:vb[3]};}
 manifest.figures.push(saveDot('all','全部88模块论文式完整图.svg','全部 88 个计算模块 · 论文式完整展开',stageDot({ids:main.nodes.map((_,i)=>i)},0,true)));
 stages.forEach((s,i)=>manifest.stages.push({...s,...saveDot(s.id,'sections/'+s.id+'.svg',s.title,stageDot(s,i))}));
 guide.sections.forEach(s=>manifest.details.push({...s,...saveDot('detail:'+s.id,'details/'+s.id+'.svg',s.title,detailDot(s))}));
 const nodes=main.nodes.map((n,i)=>({id:i,name:n.name,description:n.description,detail:guide.sections.find(s=>s.title===n.type.graph)?.id,attributes:n.attributes,stage:stageOf.get(i)}));
 fs.writeFileSync(path.join(ROOT,'paper_data.js'),'window.PAPER_DATA='+JSON.stringify({manifest,views,nodes})+';\n');fs.writeFileSync(path.join(ROOT,'manifest.json'),JSON.stringify(manifest,null,2));
 console.log(JSON.stringify({figures:manifest.figures,stages:8,details:35,nodes:88,edges:146}));
}
build().catch(e=>{console.error(e);process.exitCode=1;});
