// Lists the false positives (and optionally the dup/shifted boxes) of a bench replay, with what the core knew about each box.
// usage: node eval/bench_errors.cjs <replay dir> <seq,seq,...> [FP|DUP|SHIFT]
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..'),dir=process.argv[2],seqs=process.argv[3].split(','),want=process.argv[4]||'FP';
const {evaluate}=require(path.join(root,'viewer/evaluation.js'));
for(const seq of seqs){
 const tp=path.join(root,'viewer/data',seq,'ground-truth.json');if(!fs.existsSync(tp))continue;
 const frames=JSON.parse(fs.readFileSync(path.join(dir,seq+'.json'),'utf8')).frames,by=new Map(frames.map(f=>[f.frame,f]));
 const ev=evaluate(frames,JSON.parse(fs.readFileSync(tp,'utf8')),{pad:.2,ambiguous:'ignore',maxRange:300});
 for(const e of ev.errors.filter(e=>e.type===want)){
  const c=(by.get(e.frame).candidates||[]).find(c=>c.confirmed&&JSON.stringify(c.bbox)===JSON.stringify(e.parts[0]))||{};
  console.log(seq.padEnd(22),'frame',e.frame,'d',e.distance?.toFixed(1),c.held?`HELD${c.held_frames}`:'fresh','pts',c.point_count,'conf',c.confident_points,'track',c.track_id,e.name);
 }
}
