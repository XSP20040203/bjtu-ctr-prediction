/* Draw the documented structure only. Never imports training code or loads labels. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { instance } = require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@viz-js/viz');
const ROOT = __dirname, SOURCE = path.dirname(ROOT);
const modelText = fs.readFileSync(path.join(SOURCE, 'integrated_model.netron'), 'utf8');
const model = JSON.parse(modelText), main = model.graphs[0];
const guide = JSON.parse(fs.readFileSync(path.join(SOURCE, 'model_data.json'), 'utf8'));
const colors = {
  Data: ['#e8f2fa', '#245274'], Transform: ['#f0f3f6', '#445467'],
  Layer: ['#ececfb', '#4d4785'], Tree: ['#e8f4ed', '#2d6546'],
  Merge: ['#fff1d7', '#815f19'], Normalization: ['#fcebdc', '#87532e'],
  Control: ['#fbe9ea', '#8b3f46'], Port: ['#f3f5f8', '#647084']
};
const stages = [
  {id:'input', title:'输入对齐与特征准备', ids:[0,1,2,11], note:'四张表先对齐，再独立准备类别编号、数值特征和用户日上下文。'},
  {id:'foundation', title:'基础模型与概率融合', ids:[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21], note:'神经网络和树模型并行预测，经过加权融合与校准得到两条用户日支路共同的起点。'},
  {id:'day', title:'两条用户日并行路径', ids:Array.from({length:22},(_,i)=>22+i), note:'左路逐级校正；右路的十二棵树独立预测，经元模型合成。两路最后按差值公式汇合。'},
  {id:'response', title:'行级与用户日响应联合校正', ids:Array.from({length:8},(_,i)=>44+i), note:'行模型与用户日模型交替校正；二元响应和泊松计数在同一个父结果上分叉。'},
  {id:'quality', title:'资讯质量与并行增量迁移', ids:Array.from({length:11},(_,i)=>52+i), note:'行级质量树共用同一个锚点；稠密行级增量和紧凑日级增量分别计算后再相加。'},
  {id:'ranking', title:'排序与冷任务受众校正', ids:Array.from({length:6},(_,i)=>63+i), note:'依次调整用户内排序、用户日点击率、跨用户排序、冷任务与任务受众。'},
  {id:'history', title:'历史事件与上下文条件分支', ids:Array.from({length:9},(_,i)=>69+i), note:'先处理无序历史和应用变化事件，再从同一个父结果分出冷任务、非冷任务两路。'},
  {id:'representation', title:'冻结神经表示与最终提交', ids:Array.from({length:10},(_,i)=>78+i), note:'533、489、401维是独立构造的三份输入。三个树模型的概率校正按图中顺序执行。'}
];
const stageOf = new Map(stages.flatMap((s,i)=>s.ids.map(id=>[id,i])));
const producer = new Map();
main.inputs.forEach((p,i)=>p.value.forEach(v=>producer.set(v,'in'+i)));
main.nodes.forEach((n,i)=>n.outputs.forEach(p=>p.value.forEach(v=>producer.set(v,i))));
const edges=[];
main.nodes.forEach((n,i)=>n.inputs.forEach((p,j)=>p.value.forEach(v=>edges.push({from:producer.get(v),to:i,value:v,input:j}))));
const outputs=main.outputs.flatMap(p=>p.value.map(v=>({from:producer.get(v),to:'out',value:v})));
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const dq=s=>JSON.stringify(String(s));
const textUnits=s=>[...s].reduce((a,c)=>a+(c.charCodeAt(0)>255?1:0.53),0);
function wrap(s,max=28){
  const lines=[];
  for(const raw of String(s).split('\n')){
    let line='',units=0;
    const tokens=raw.match(/[A-Za-z0-9_]+(?:[.][0-9]+)*|[^A-Za-z0-9_]/gu)||[''];
    for(const token of tokens){
      const size=textUnits(token);
      if(line&&units+size>max){lines.push(line.trim());line='';units=0;}
      line+=token;units+=size;
    }
    lines.push(line.trim());
  }
  return lines;
}
const br=(s,max)=>wrap(s,max).map(esc).join('<BR ALIGN="LEFT"/>');
const dim=v=>((main.values[v]?.type?.shape?.dimensions)||[]).join(' × ');
const symbolic = {
  calibrated:'p_cal',day_count:'p_day',previous_day:'p_prev',ranked:'p_ranked',
  joined:'p_join',protected_weighted:'p_weighted',protected_rank:'p_rank',
  quality_anchor:'p_anchor',group_adjusted:'p_group',quality_relative:'p_quality'
};
function outputLine(n){
  return n.outputs.flatMap(o=>o.value.map(v=>{
    const value=main.values[v];return '输出：'+dim(v)+(symbolic[value.name]?'  '+symbolic[value.name]:'');
  })).join('；');
}
function label(title,attrs=[],category='Data',footer='',small=false){
  const [fill,ink]=colors[category]||colors.Transform;
  let rows=`<TR><TD ALIGN="LEFT" BGCOLOR="${fill}" CELLPADDING="11"><FONT COLOR="${ink}" POINT-SIZE="16"><B>${br(title,25)}</B></FONT></TD></TR>`;
  for(const a of attrs)rows+=`<TR><TD ALIGN="LEFT" CELLPADDING="7"><FONT POINT-SIZE="13.5">${br(a,29)}</FONT></TD></TR>`;
  if(footer)rows+=`<TR><TD ALIGN="LEFT" BGCOLOR="#f7f9fb" CELLPADDING="8"><FONT COLOR="#516174" POINT-SIZE="12.5">${br(footer,29)}</FONT></TD></TR>`;
  return `<<TABLE BORDER="1" COLOR="#bac7d4" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" WIDTH="${small?280:390}" BGCOLOR="#ffffff">${rows}</TABLE>>`;
}
function node(i){
  const n=main.nodes[i],num=String(i+1).padStart(2,'0');
  const a=n.attributes.map(a=>a.name+'：'+a.value);
  if(n.type.category==='Control'){
    const lines=[...wrap(num+'  '+n.name,23),...a.flatMap(s=>wrap(s,25)),outputLine(n)];
    return `n${i} [id="node-${i}" shape=diamond style=filled fillcolor="#fbe9ea" color="#b28a91" fontcolor="#713a43" fontsize=14 margin="0.25,0.14" label=${dq(lines.join('\n'))}];`;
  }
  return `n${i} [id="node-${i}" tooltip=${dq(n.description||n.name)} label=${label(num+'  '+n.name,a,n.type.category,outputLine(n))}];`;
}
function compactParallel(s){
  // Invisible placement constraints only; every visible arrow remains a real dependency.
  const rows=s.id==='foundation'?[[3,4,5,6],[7,8,9,10],[12,13,14,15]]:
    s.id==='day'?[[26,27,28,29],[30,31,32,33],[34,35,36,37]]:[];
  let result='';
  for(const row of rows)result+='{rank=same; '+row.map(i=>'n'+i).join('; ')+';}\n';
  for(let r=1;r<rows.length;r++)for(let c=0;c<rows[r].length;c++)result+=`n${rows[r-1][c]} -> n${rows[r][c]} [style=invis,weight=50];\n`;
  return result;
}
function graphHead(title,subtitle){
  return `digraph model {\ngraph [rankdir=TB, bgcolor="#ffffff", pad=0.45, nodesep=0.38, ranksep=0.7, splines=polyline, outputorder=edgesfirst, fontname="Microsoft YaHei", labeljust=l, labelloc=t, compound=true, newrank=true, label=<<FONT POINT-SIZE="26"><B>${esc(title)}</B></FONT><BR/><FONT POINT-SIZE="14">${esc(subtitle)}</FONT>>];\nnode [shape=plain,fontname="Microsoft YaHei"];\nedge [color="#78899b",arrowsize=0.75,penwidth=1.25,fontname="Microsoft YaHei",fontsize=12,fontcolor="#4c6077"];\n`;
}
function edgeLabel(e){
  const weights={16:['0.24','0.24','0.12','0.20','0.20'],17:['0.675','0.225','0.10'],19:['0.504','0.126','0.07','0.30'],20:['0.75','0.25'],48:['0.75','0.25']};
  if(weights[e.to])return weights[e.to][e.input];
  const val=main.values[e.value];
  if(symbolic[val.name])return symbolic[val.name];
  if([4,5,6,15,82,83,84,85,86].includes(e.value))return dim(e.value);
  return '';
}
function edge(e,from,to,extra=''){
  const l=edgeLabel(e);
  return `${from||('n'+e.from)} -> ${to||('n'+e.to)} [${l?'label='+dq(l)+',':''} ${extra}];`;
}
function mainDot(){
  let dot=graphHead('整合点击率模型 · 完整结构流程图','N = 404,182 条广告；G = 8,556 个用户日；T = 34,510 个任务日；编号只表示图中模块顺序');
  for(let i=0;i<main.inputs.length;i++)dot+=`in${i} [id="input-${i}" label=${label(main.inputs[i].name,[], 'Data','输入表')}];\n`;
  for(let j=0;j<stages.length;j++){
    const s=stages[j];
    dot+=`subgraph cluster_${s.id} { label=${dq(String(j+1).padStart(2,'0')+'  '+s.title)}; fontsize=22; fontcolor="#26394c"; color="#cad4df"; style="rounded"; margin=24;\n`;
    for(const id of s.ids)dot+=node(id)+'\n';
    dot+=compactParallel(s);
    dot+='}\n';
  }
  for(const e of edges)dot+=edge(e,typeof e.from==='string'?e.from:null)+'\n';
  dot+=`out [id="output-final" label=${label('提交文件', ['id、pctr 两列','404,182 行；与原始预测表逐行对应'], 'Data','最终概率 N × 1')}];\nn87 -> out;\n}`;
  return dot;
}
function stageDot(s,index){
  const ids=new Set(s.ids),inPorts=new Map(),outPorts=new Map();
  for(const e of edges){
    if(ids.has(e.to)&&!ids.has(e.from))inPorts.set(e.from,e.value);
    if(ids.has(e.from)&&!ids.has(e.to))outPorts.set(e.from,e.value);
  }
  let dot=graphHead(String(index+1).padStart(2,'0')+'  '+s.title,s.note);
  for(const id of s.ids)dot+=node(id)+'\n';
  dot+=compactParallel(s);
  for(const [id,v]of inPorts){
    const input=typeof id==='string',name=input?main.inputs[Number(id.slice(2))].name:main.nodes[id].name;
    const origin=input?'原始输入表':`来自第 ${stageOf.get(id)+1} 部分 · 模块 ${id+1}`;
    dot+=`portIn_${id} [id="from-${id}" label=${label(origin,[name], 'Port',dim(v),true)}];\n`;
  }
  for(const [id,v]of outPorts){
    const destinations=[...new Set(edges.filter(e=>e.from===id&&!ids.has(e.to)).map(e=>stageOf.get(e.to)+1))];
    dot+=`portOut_${id} [id="to-${id}" label=${label('继续送入第 '+destinations.join(' / ')+' 部分',[main.nodes[id].name], 'Port',dim(v),true)}];\n`;
  }
  for(const e of edges){
    if(ids.has(e.to))dot+=edge(e,ids.has(e.from)?'n'+e.from:'portIn_'+e.from)+'\n';
  }
  for(const [id]of outPorts)dot+=`n${id} -> portOut_${id} [style=dashed,color="#9ba9b8"];\n`;
  if(ids.has(87)){
    dot+=`out [id="output-final" label=${label('最终提交文件',['id、pctr 两列','按原始行序输出 404,182 行'], 'Data','N × 1 点击概率')}];\nn87 -> out;\n`;
  }
  return dot+'}';
}
function detailDot(s,index){
  let dot=graphHead(s.title, '内部结构详图 · 所有维度按样本批次 B、广告行 N、用户日 G 或任务日 T 标注');
  s.nodes.forEach((n,i)=>{
    let footer='';
    if(s.id==='embedding'&&i===2)footer='在同一个成员内部共享；不同网络成员的嵌入权重各自独立。';
    const title=n.title;
    const body=n.body.split('\n');
    dot+=`d${i} [id="detail-node-${i}" label=${label(title,body,(/树|CatBoost|LightGBM/.test(title)?'Tree':/融合|拼接|元模型/.test(title)?'Merge':'Layer'),footer)}];\n`;
  });
  for(const row of s.rows)if(row.length>1)dot+='{rank=same; '+row.map(i=>'d'+i).join('; ')+';}\n';
  for(const [a,b]of s.edges)dot+=`d${a} -> d${b};\n`;
  return dot+'}';
}
function cleanSvg(svg,title,desc){
  return svg.replace(/<\?xml[^>]*>\s*/,'').replace(/<!DOCTYPE[\s\S]*?>\s*/,'')
    .replace('<svg ', '<svg role="img" aria-label="'+esc(title)+'" ')
    .replace(/(<svg[^>]*>)/,'$1\n<desc>'+esc(desc)+'</desc>')
    .replace(/font-family="Microsoft YaHei"/g,'font-family="Microsoft YaHei, SimHei, sans-serif"');
}
async function run(){
  fs.mkdirSync(path.join(ROOT,'sections'),{recursive:true});
  fs.mkdirSync(path.join(ROOT,'details'),{recursive:true});
  fs.mkdirSync(path.join(ROOT,'qa'),{recursive:true});
  const viz=await instance();
  const manifest={sourceSha256:crypto.createHash('sha256').update(modelText).digest('hex'),nodes:88,inputs:4,outputs:1,edges:edges.length+1,stages:[],details:[]};
  function render(dot,file,title,desc){
    const result=viz.render(dot,{format:'svg'});
    if(result.status!=='success')throw Error(JSON.stringify(result.errors));
    if(result.errors?.length)console.log('layout diagnostics',file,result.errors);
    const svg=cleanSvg(result.output,title,desc);
    fs.writeFileSync(path.join(ROOT,file),svg,'utf8');
    const vb=svg.match(/viewBox="([^"]+)"/)[1].split(' ').map(Number);
    return {file,width:vb[2],height:vb[3]};
  }
  manifest.full=render(mainDot(),'完整结构总流程图.svg','完整结构总流程图','四张输入表，88个计算模块，完整依赖关系与最终pctr输出。每个模块框内显示特征维度、参数与融合公式。');
  console.log('full',manifest.full);
  for(let i=0;i<stages.length;i++){
    const s=stages[i],f='sections/'+String(i+1).padStart(2,'0')+'-'+s.id+'.svg';
    manifest.stages.push({...s,...render(stageDot(s,i),f,s.title,s.note)});
  }
  for(let i=0;i<guide.sections.length;i++){
    const s=guide.sections[i],f='details/'+s.id+'.svg';
    manifest.details.push({id:s.id,title:s.title,paragraphs:s.paragraphs,...render(detailDot(s,i),f,s.title,s.paragraphs.join('\n'))});
  }
  // Inline the vectors so the delivered HTML also works when opened by double-click.
  const svgs={full:fs.readFileSync(path.join(ROOT,manifest.full.file),'utf8')};
  for(const s of manifest.stages)svgs[s.id]=fs.readFileSync(path.join(ROOT,s.file),'utf8');
  for(const s of manifest.details)svgs['detail:'+s.id]=fs.readFileSync(path.join(ROOT,s.file),'utf8');
  const nodes=main.nodes.map((n,i)=>({id:i,name:n.name,description:n.description,detail:guide.sections.find(s=>s.title===n.type.graph)?.id,attributes:n.attributes,stage:stageOf.get(i)}));
  fs.writeFileSync(path.join(ROOT,'flowchart_data.js'),'window.FLOWCHART_DATA = '+JSON.stringify({manifest,svgs,nodes})+';\n','utf8');
  fs.writeFileSync(path.join(ROOT,'manifest.json'),JSON.stringify(manifest,null,2),'utf8');
  console.log(JSON.stringify({full:manifest.full,stages:manifest.stages.map(s=>({id:s.id,w:s.width,h:s.height})),details:manifest.details.length,nodes:manifest.nodes,edges:manifest.edges}));
}
run().catch(e=>{console.error(e);process.exitCode=1;});
