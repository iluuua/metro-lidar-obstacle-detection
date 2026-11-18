// Scores saved core replays (eval/bench.py) against the markup, with the viewer's own evaluator.
// usage: node eval/bench_score.cjs <replay dir> <seq,seq,...>   -> JSON on stdout
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..'),dir=process.argv[2],seqs=process.argv[3].split(',');
const {evaluate}=require(path.join(root,'viewer/evaluation.js'));
const controls=new Set((process.argv[4]||'').split(',').filter(Boolean));
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const round=(v,n=1)=>v==null?null:+v.toFixed(n);
const out={};
for(const seq of seqs){
 const frames=read(path.join(dir,seq.replace(':','-')+'.json')).frames,base=seq.split('@')[0];
 const alarms=frames.filter(f=>f.obstacle).length,attention=frames.filter(f=>!f.obstacle&&f.status==='candidate').length;
 const truthPath=path.join(root,'viewer/data',base,'ground-truth.json'),truth=fs.existsSync(truthPath)?read(truthPath):null;
 const labelled=truth&&truth.frames.some(f=>f.objects.some(o=>o.label==='positive'));
 if(!labelled){
  // clean control (or the real recording with a person on the track: alarm frames are then the point)
  let first=null;for(const f of frames)if(f.obstacle){const d=(f.candidates||[]).filter(c=>c.confirmed).map(c=>c.distance_m);first=d.length?Math.max(...d.filter(Number.isFinite)):null;break}
  out[seq]={kind:controls.has(seq)||controls.has(base)?'control':'unlabelled',alarm_frames:alarms,attention_frames:attention,frames:frames.length,first_alarm_distance:round(first),
    first_alarm_frame:frames.find(f=>f.obstacle)?.frame??null};
  continue;
 }
 const ev=evaluate(frames,truth,{pad:.2,ambiguous:'ignore',maxRange:300});
 const objects=ev.objects.map(o=>({id:o.id,visible:o.visible,tp:o.tp,shifted:o.shifted,fn:o.fn,first_distance:round(o.firstAnyDistance),
   coverage_after_first:o.afterFirst?+(o.matchedAfterFirst/o.afterFirst).toFixed(3):null,longest_gap_frames:o.gapMax}));
 out[seq]={kind:'labelled',counts:ev.counts,found:`${ev.detected}/${ev.total}`,objects,alarm_frames:alarms,attention_frames:attention,
   disputed:{visible:ev.boundary.counts.visible,confirmed:ev.boundary.counts.confirmed,attention:ev.boundary.counts.attention,missed:ev.boundary.counts.missed,falseAttention:ev.boundary.counts.falseAttention}};
}
process.stdout.write(JSON.stringify(out));
