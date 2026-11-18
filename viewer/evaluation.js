/* Object-level, per-frame evaluation; no detector geometry used to label truth. */
(function(root){
 'use strict';
 const intersects=(a,b,pad=0)=>a?.length===2&&b?.length===2&&[0,1,2].every(k=>a[1][k]>=b[0][k]-pad&&a[0][k]<=b[1][k]+pad);
 // Largest gap between a confirmed box and a target that still counts as "the target is alarmed".
 const NEAR=1.5;
 const boxGap=(a,b)=>Math.hypot(...[0,1,2].map(k=>Math.max(0,a[0][k]-b[1][k],b[0][k]-a[1][k])));
 const isAttention=c=>c.attention??Boolean(c.boundary||c.far_attention||c.confident_points>0||c.appeared_frac>0);
 function boundaryEvaluation(frames,truth,{pad,minRange,maxRange}){
  const byId=new Map(frames.map(f=>[f.frame,f])),objects=new Map(),errors=[];
  const counts={visible:0,covered:0,confirmed:0,attention:0,missed:0,falseAttention:0};
  for(const gt of truth.frames){
   const frame=byId.get(gt.frame)||{candidates:[]},inRange=o=>o.parts.some(b=>b[1][0]>=minRange&&b[0][0]<=maxRange);
   const disputed=gt.objects.filter(o=>o.label==='ambiguous'&&inRange(o)),regular=gt.objects.filter(o=>o.label==='positive'&&inRange(o));
   const alerts=(frame.candidates||[]).filter(c=>(c.confirmed||isAttention(c))&&c.bbox?.length===2&&c.bbox[1][0]>=minRange&&c.bbox[0][0]<=maxRange);
   const overlaps=(c,o)=>o.parts.some(b=>intersects(c.bbox,b,pad)),match=new Map();
   function assign(pi,seen){for(let oi=0;oi<disputed.length;oi++){if(seen.has(oi)||!overlaps(alerts[pi],disputed[oi]))continue;seen.add(oi);if(!match.has(oi)||assign(match.get(oi),seen)){match.set(oi,pi);return true}}return false}
   alerts.forEach((_,i)=>assign(i,new Set()));
   disputed.forEach((o,oi)=>{
    const stat=objects.get(o.id)||{id:o.id,name:o.name,visible:0,covered:0,confirmed:0,attention:0,missed:0,longestMiss:0,missRun:0,firstFrame:null,firstDistance:null};
    if(stat.lastFrame!=null&&gt.frame-stat.lastFrame!==1)stat.missRun=0;
    stat.lastFrame=gt.frame;stat.visible++;counts.visible++;
    if(match.has(oi)){
     const c=alerts[match.get(oi)],kind=c.confirmed?'confirmed':'attention';stat.covered++;stat[kind]++;counts.covered++;counts[kind]++;stat.missRun=0;
     if(stat.firstFrame===null){stat.firstFrame=gt.frame;stat.firstDistance=Math.min(...o.parts.map(b=>b[0][0]))}
    }else{
     stat.missed++;counts.missed++;stat.missRun++;stat.longestMiss=Math.max(stat.longestMiss,stat.missRun);
     errors.push({type:'WFN',key:o.id,name:o.name,frame:gt.frame,time_s:gt.time_s,parts:o.parts,distance:Math.min(...o.parts.map(b=>b[0][0]))});
    }
    objects.set(o.id,stat);
   });
   const matched=new Set(match.values());
   alerts.forEach((c,i)=>{if(c.confirmed||matched.has(i)||regular.some(o=>overlaps(c,o)))return;counts.falseAttention++;errors.push({type:'WFP',key:String(c.track_id??i),name:'Внимание без соответствия разметке',frame:gt.frame,time_s:gt.time_s,parts:[c.bbox],distance:c.distance_m})});
  }
  return {counts,objects:[...objects.values()],errors};
 }
 function evaluate(frames,truth,{pad=.2,ambiguous='ignore',minRange=1,maxRange=300}={}){
  const byId=new Map(frames.map(f=>[f.frame,f])),counts={tp:0,fp:0,fn:0,fnBeyond:0,fpExterior:0,fpStale:0,dup:0,shifted:0,ignored:0,attention:0,missingFrames:0,evaluatedFrames:0},errors=[],perObject=new Map(),perFrame=new Map();
  for(const gt of truth.frames){
   let frame=byId.get(gt.frame);if(!frame){counts.missingFrames++;frame={frame:gt.frame,status:'missing',candidates:[]}}
   counts.evaluatedFrames++;
   const objects=gt.objects.filter(o=>o.parts.some(b=>b[1][0]>=minRange&&b[0][0]<=maxRange));
   const positives=objects.filter(o=>o.label==='positive'||o.label==='ambiguous'&&ambiguous==='positive');
   const ignored=objects.filter(o=>o.label==='ambiguous'&&ambiguous==='ignore');
   const confirmed=(frame.candidates||[]).map((c,index)=>({...c,index})).filter(c=>c.confirmed&&c.bbox?.length===2&&c.bbox[1][0]>=minRange&&c.bbox[0][0]<=maxRange);
   const overlaps=(c,o)=>o.parts.some(b=>intersects(c.bbox,b,pad));
   const near=(c,o)=>o.parts.some(p=>boxGap(c.bbox,p)<=NEAR);
   // Maximum-cardinality bipartite matching: one prediction cannot satisfy two objects.
   function matching(predictions,targets,fits){
    const found=new Map();
    function assign(pi,seen){
     for(const oi of targets){
      if(seen.has(oi)||!fits(confirmed[pi],positives[oi]))continue;
      seen.add(oi);
      if(!found.has(oi)||assign(found.get(oi),seen)){found.set(oi,pi);return true}
     }
     return false;
    }
    predictions.forEach(pi=>assign(pi,new Set()));
    return found;
   }
   const every=confirmed.map((_,i)=>i),targets=positives.map((_,i)=>i),match=matching(every,targets,overlaps);
   // What the strict rule leaves over: a box still next to a target is that target alarmed with a
   // shifted (usually held) box. It is neither a miss nor a false alarm; it is counted on its own.
   const strict=new Set(match.values()),shifted=matching(every.filter(i=>!strict.has(i)),targets.filter(i=>!match.has(i)),near);
   const matched=new Set([...strict,...shifted.values()]),row={tp:match.size,fp:0,fn:0,fnBeyond:0,fpStale:0,dup:0,shifted:shifted.size,errors:[]};
   // A miss beyond the range the detector itself claims to have checked is a different kind of miss.
   const verified=Number.isFinite(frame.analysis_range_m)?frame.analysis_range_m:0;
   counts.tp+=match.size;
   positives.forEach((obj,oi)=>{
    const stats=perObject.get(obj.id)||{id:obj.id,name:obj.name,visible:0,tp:0,shifted:0,fn:0,firstFrame:null,firstDistance:null,firstAnyFrame:null,firstAnyDistance:null,afterFirst:0,matchedAfterFirst:0,gapRun:0,gapMax:0};
    stats.visible++;
    const distance=Math.min(...obj.parts.map(b=>b[0][0]));
    if(match.has(oi)){
     stats.tp++;
     if(stats.firstFrame===null){stats.firstFrame=gt.frame;stats.firstDistance=distance}
    }else if(shifted.has(oi)){
     const c=confirmed[shifted.get(oi)];stats.shifted++;counts.shifted++;
     const error={type:'SHIFT',key:obj.id,name:obj.name,frame:gt.frame,time_s:gt.time_s,parts:[c.bbox],distance:c.distance_m??distance,held:Boolean(c.held)};errors.push(error);row.errors.push(error);
    }else{
     stats.fn++;counts.fn++;row.fn++;
     const beyond=distance>verified;
     if(beyond){counts.fnBeyond++;row.fnBeyond++}
     const error={type:'FN',key:obj.id,name:obj.name,frame:gt.frame,time_s:gt.time_s,parts:obj.parts,distance,beyondVerified:beyond};errors.push(error);row.errors.push(error);
    }
    // first detection with a shifted box counted, and how steadily the object is held after it
    const hit=match.has(oi)||shifted.has(oi);
    if(hit&&stats.firstAnyFrame===null){stats.firstAnyFrame=gt.frame;stats.firstAnyDistance=distance}
    if(stats.firstAnyFrame!==null){stats.afterFirst++;if(hit){stats.matchedAfterFirst++;stats.gapRun=0}else{stats.gapRun++;stats.gapMax=Math.max(stats.gapMax,stats.gapRun)}}
    perObject.set(obj.id,stats);
   });
   confirmed.forEach((c,i)=>{
    if(matched.has(i))return;
    // Every target near this box is matched by now: this is a second box on a target that is already alarmed.
    const twin=positives.find(o=>near(c,o));
    if(twin){
     counts.dup++;row.dup++;
     const error={type:'DUP',key:String(c.track_id??c.index),name:twin.name,frame:gt.frame,time_s:gt.time_s,parts:[c.bbox],distance:c.distance_m,held:Boolean(c.held)};errors.push(error);row.errors.push(error);return;
    }
    if(ignored.some(o=>near(c,o))){counts.ignored++;return}
    counts.fp++;row.fp++;
    const exterior=objects.find(o=>(o.label==='negative'||o.label==='ambiguous'&&ambiguous==='negative')&&overlaps(c,o));
    // A held box with no point left in it is a leftover of an earlier detection, not something seen now.
    const stale=Boolean(c.held)&&!(c.point_count>0);
    if(exterior)counts.fpExterior++;
    if(stale){counts.fpStale++;row.fpStale++}
    const error={type:'FP',key:String(c.track_id??c.index),name:exterior?exterior.name:'Без соответствия разметке',exterior:Boolean(exterior),held:Boolean(c.held),stale,points:c.point_count??null,frame:gt.frame,time_s:gt.time_s,
      parts:[c.bbox],distance:c.distance_m};errors.push(error);row.errors.push(error);
   });
   if((frame.candidates||[]).some(c=>!c.confirmed&&isAttention(c)))counts.attention++;
   perFrame.set(gt.frame,row);
  }
  const boundary=boundaryEvaluation(frames,truth,{pad,minRange,maxRange});
  errors.push(...boundary.errors);errors.sort((a,b)=>a.frame-b.frame);
  for(const e of boundary.errors){const row=perFrame.get(e.frame);if(row){row.errors.push(e);row[e.type==='WFN'?'warningMiss':'warningFalse']=(row[e.type==='WFN'?'warningMiss':'warningFalse']||0)+1}}
  const episodes=[],latest=new Map();
  for(const e of errors){
   const key=`${e.type}:${e.key}`,old=latest.get(key);
   if(old&&e.frame-old.endFrame===1){old.endFrame=e.frame;old.endTime=e.time_s;old.count++;if((e.distance??Infinity)<(old.example.distance??Infinity))old.example=e}
   else{const episode={type:e.type,key:e.key,name:e.name,startFrame:e.frame,endFrame:e.frame,startTime:e.time_s,endTime:e.time_s,count:1,example:e};episodes.push(episode);latest.set(key,episode)}
  }
  const objects=Array.from(perObject.values());
  return {counts,errors,episodes,perFrame,objects,boundary,detected:objects.filter(o=>o.tp>0||o.shifted>0).length,total:objects.length};
 }
 const api={intersects,evaluate,isAttention,NEAR};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.DetectorEvaluation=api;
})(typeof window!=='undefined'?window:this);
