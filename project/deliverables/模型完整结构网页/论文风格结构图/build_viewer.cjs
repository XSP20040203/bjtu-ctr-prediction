const fs=require('fs'),path=require('path');
const old=path.join(__dirname,'..','完整流程图');
let html=fs.readFileSync(path.join(old,'index.html'),'utf8');
html=html.replaceAll('整合模型完整流程图','整合模型论文式结构图')
 .replace('从四张输入表到最终 pctr · 88 个计算模块 · 参数、维度和融合公式直接写在框内','论文式总图 · 核心网络张量展开 · 冻结关系表示 · 全部 88 个模块与参数')
 .replaceAll('完整结构总流程图.svg','论文式模型整体结构图.svg').replaceAll('完整结构总流程图.png','论文式模型整体结构图.png')
 .replace('保存完整矢量图','保存论文主图 SVG').replace('保存完整 PNG','保存论文主图 PNG')
 .replace('<h2>完整结构</h2><button class="nav-item" data-view="full">完整总流程图 · 全部 88 个模块</button>',
 '<h2>论文主图与模块展开</h2><button class="nav-item" data-view="paper">图 1 · 整合模型总结构</button><button class="nav-item" data-view="neural">图 2 · 六个核心网络展开</button><button class="nav-item" data-view="final">图 3 · 冻结表示与条件融合</button><button class="nav-item" data-view="all">全部 88 个模块 · 完整参数</button>')
 .replace('总图保留全部连接。想看清文字，可点左侧的 8 段流程，或用滚轮放大；点击计算模块可继续展开内部结构。','图 1 为模块化总图：同名 p_cal、p_rank、p_parent 端口表示同一份分数。点击分组看内部连线；全部 88 模块图保留完整依赖。滚轮缩放，拖动平移。')
 .replace('flowchart_data.js','paper_data.js').replace('viewer.js','paper_viewer.js')
 .replace('</style>','.paper-block[data-view],.paper-block[id]{cursor:pointer}.paper-block[data-view]:hover>rect{stroke:#315e80;stroke-width:2.8}</style>');
fs.writeFileSync(path.join(__dirname,'index.html'),html);
let viewer=fs.readFileSync(path.join(old,'viewer.js'),'utf8');
viewer=viewer.replace('const {manifest, svgs, nodes}=window.FLOWCHART_DATA;','const {manifest, views:svgs, nodes}=window.PAPER_DATA;')
 .replace("let selected='full'","let selected='paper'")
 .replace("const views=[{id:'full',title:'完整总流程图',...manifest.full},...manifest.stages,...manifest.details.map(s=>({...s,id:'detail:'+s.id}))];","const views=[...manifest.figures,...manifest.stages,...manifest.details];")
 .replace("o.value='detail:'+s.id","o.value=s.id")
 .replace("if(!svgs[id])id='full'","if(!svgs[id])id='paper'")
 .replace("selected==='full'","selected==='all'")
 .replace("||'full',false","||'paper',false")
 .replace("const g=e.target.closest('g.node');if(!g)return;","const g=e.target.closest('g.node,g.paper-block');if(!g)return;if(g.dataset.view){choose(g.dataset.view);return;}");
fs.writeFileSync(path.join(__dirname,'paper_viewer.js'),viewer);
console.log('Created paper viewer.');
