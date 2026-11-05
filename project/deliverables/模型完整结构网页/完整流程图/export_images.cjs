const fs=require('fs'),path=require('path');
const sharp=require('C:/Users/30272/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const root=__dirname,manifest=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
(async()=>{
 const all=[manifest.full,...manifest.stages,...manifest.details];
 for(const item of all){
   const full=item===manifest.full;
   const w=full?6000:Math.min(4200,Math.round(item.width*1.7));
   const h=Math.round(item.height*w/item.width);
   // Rasterize vectors with the Windows font collection, including Microsoft YaHei.
   const input=fs.readFileSync(path.join(root,item.file));
   await sharp(input,{density:full?72:110,limitInputPixels:220000000}).resize({width:w,height:h,fit:'inside'}).png({compressionLevel:9}).toFile(path.join(root,item.file.replace(/\.svg$/,'.png')));
   if(full)console.log('Full PNG:',w,h);
 }
 console.log('Exported PNG files:',all.length);
})().catch(e=>{console.error(e);process.exitCode=1;});
