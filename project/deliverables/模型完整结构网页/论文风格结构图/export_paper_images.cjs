const fs=require('fs'),path=require('path');
const sharp=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const root=__dirname,m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
(async()=>{
 const wanted=process.argv.slice(2);
 const all=[...m.figures,...m.stages,...m.details].filter(a=>!wanted.length||wanted.includes(a.id));
 for(const a of all){
  const handmade=['paper','neural','final'].includes(a.id);
  const w=handmade?a.width*2:Math.min(a.id==='all'?5200:4200,Math.round(a.width*1.8));
  try{
   await sharp(fs.readFileSync(path.join(root,a.file)),{density:handmade?144:a.id==='all'?72:110,limitInputPixels:220000000}).resize({width:Math.round(w)}).png({compressionLevel:9}).toFile(path.join(root,a.file.replace(/\.svg$/,'.png')));
  }catch(e){throw Error(a.id+' '+a.width+'×'+a.height+': '+e.message);}
  if(handmade)console.log('Exported',a.id,w);
 }
 console.log('PNG files:',all.length);
})().catch(e=>{console.error(e);process.exitCode=1;});
