/* Lane data for the multi-level timeline: one cell per frame and lane, no DOM. */
(function(root){
 'use strict';
 const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 const STRAIGHT=1/3000;
 // Lane order top to bottom. A larger code wins when several frames share one pixel.
 const LANES=[
  {id:'decision',label:'Решение',hint:'Что выдал алгоритм',codes:{1:['#2f7a6d','Путь свободен'],2:['#6c7480','Нет решения'],3:['#e0a85d','Внимание'],4:['#ff767c','Препятствие']}},
  {id:'truth',label:'Разметка',hint:'Что объявил автор записи',codes:{1:['#4a5663','Только внешние объекты и пыль'],2:['#b48dff','Спорный объект'],3:['#4fc3f7','Целевой объект']}},
  {id:'attention',label:'Внимание',hint:'Сигнал без подтверждения',codes:{1:['#f4bc73','Внимание']}},
  {id:'axis',label:'Ось',hint:'Ось пути изгибается обратно',codes:{1:['#e8b76b','За концом данных'],2:['#ff5b61','Внутри измеренной части']},step:true},
  {id:'fp',label:'FP',hint:'Подтверждённая тревога без цели в разметке',codes:{1:['#ff767c','Ложная тревога, рамка с точками'],2:['#a8484e','Ложная тревога, удержанная рамка без точек']},step:true,errors:['FP']},
  {id:'fn',label:'FN',hint:'Цель в разметке, тревоги нет',codes:{1:['#7f62b8','Пропуск за проверенной дальностью'],2:['#bc9bff','Пропуск внутри проверенной дальности']},step:true,errors:['FN']},
  {id:'near',label:'Рамка у цели',hint:'Тревога на цели, но рамка сдвинута или продублирована: не FP и не FN',codes:{1:['#8fa7ff','Вторая рамка на уже найденной цели'],2:['#37c8d4','Рамка сдвинута относительно цели']},step:true,errors:['DUP','SHIFT']},
  {id:'disputed',label:'Спорные',hint:'Спорные кубы: пропуск или лишнее внимание',codes:{1:['#d6b84a','Лишнее внимание'],2:['#ffd166','Пропуск спорного куба']},step:true,errors:['WFN','WFP']}
 ];
 /* Heading change (rad) an axis undoes by bending back; the same measure as eval/axis_reversal.py. */
 function reversal(x,y){
  if(x.length<4)return 0;
  const psi=[],len=[];
  for(let i=1;i<x.length;i++){const dx=x[i]-x[i-1],dy=y[i]-y[i-1];psi.push(Math.atan2(dy,dx));len.push(Math.hypot(dx,dy))}
  let pos=0,neg=0;
  for(let i=1;i<psi.length;i++){
   const mid=Math.max(.5*(len[i]+len[i-1]),1e-6);let turn=psi[i]-psi[i-1];
   if(turn>Math.PI)turn-=2*Math.PI;if(turn<-Math.PI)turn+=2*Math.PI;
   if(Math.abs(turn/mid)<STRAIGHT)continue;
   if(turn>0)pos+=turn;else neg-=turn;
  }
  return 2*Math.min(pos,neg);
 }
 /* {inside, beyond} reversal in rad, relative to the measured support, or null without an axis. */
 function axisReversal(frame){
  const path=frame?.path;
  if(!Array.isArray(path)||path.length<6||frame.status==='unknown')return null;
  const support=Math.max(+frame.path_info?.support_m||0,30),inside=[[],[]],beyond=[[],[]];
  for(const point of path){const side=point[0]<=support?inside:null;if(side){side[0].push(point[0]);side[1].push(point[1])}if(point[0]>=support){beyond[0].push(point[0]);beyond[1].push(point[1])}}
  return {inside:reversal(inside[0],inside[1]),beyond:beyond[0].length>=4?reversal(beyond[0],beyond[1]):0};
 }
 const decisionCode=f=>f.obstacle?4:f.status==='candidate'?3:f.status==='unknown'||f.status==='skipped'?2:1;
 /* cells[lane] = Uint8Array(frames), plus the indices of frames that have anything in the lane. */
 function buildLanes({frames,truth,evaluation,thresholdDeg=5}){
  const n=frames.length,cells={},indices={};
  for(const lane of LANES)cells[lane.id]=new Uint8Array(n);
  const index=new Map(frames.map((f,i)=>[f.frame,i])),limit=thresholdDeg*Math.PI/180;
  frames.forEach((f,i)=>{
   cells.decision[i]=decisionCode(f);
   if((f.candidates||[]).some(c=>!c.confirmed&&(c.attention??Boolean(c.boundary||c.far_attention||c.confident_points>0||c.appeared_frac>0))))cells.attention[i]=1;
   const r=axisReversal(f);if(r)cells.axis[i]=r.inside>limit?2:r.beyond>limit?1:0;
  });
  if(truth)for(const gt of truth.frames){
   const i=index.get(gt.frame);if(i===undefined)continue;
   cells.truth[i]=gt.objects.some(o=>o.label==='positive')?3:gt.objects.some(o=>o.label==='ambiguous')?2:gt.objects.length?1:0;
  }
  if(evaluation)for(const [id,row] of evaluation.perFrame){
   const i=index.get(id);if(i===undefined)continue;
   if(row.fp)cells.fp[i]=row.fp>row.fpStale?1:2;
   if(row.dup||row.shifted)cells.near[i]=row.shifted?2:1;
   if(row.fn)cells.fn[i]=row.fn>row.fnBeyond?2:1;
   if(row.warningFalse||row.warningMiss)cells.disputed[i]=row.warningMiss?2:1;
  }
  for(const lane of LANES)indices[lane.id]=Uint8Array.from(cells[lane.id]).reduce((a,c,i)=>(c?a.push(i):0,a),[]);
  return {cells,indices,n};
 }
 /* One code per pixel column of [start,end) squeezed into `width` columns; the strongest code wins. */
 function paint(cells,start,end,width){
  const out=new Uint8Array(width),span=end-start;
  if(span<=0||width<=0)return out;
  for(let i=Math.max(0,Math.floor(start));i<Math.min(cells.length,Math.ceil(end));i++){
   const c=cells[i];if(!c)continue;
   const x0=Math.floor((i-start)/span*width),x1=Math.max(x0+1,Math.ceil((i+1-start)/span*width));
   for(let x=Math.max(0,x0);x<Math.min(width,x1);x++)if(c>out[x])out[x]=c;
  }
  return out;
 }
 /* Closest flagged index to `target` within `tolerance` frames, or null. `indices` is ascending. */
 function nearest(indices,target,tolerance=Infinity){
  let lo=0,hi=indices.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(indices[mid]<target)lo=mid+1;else hi=mid}
  let best=null;
  for(const k of [lo-1,lo])if(k>=0&&k<indices.length&&(best===null||Math.abs(indices[k]-target)<Math.abs(best-target)))best=indices[k];
  return best!==null&&Math.abs(best-target)<=tolerance?best:null;
 }
 /* Start of the next (+1) or previous (-1) run of flagged frames, or null at the ends. */
 function runStarts(indices){return indices.filter((v,k)=>k===0||v-indices[k-1]>1)}
 function step(indices,current,direction){
  const starts=runStarts(indices);
  if(direction>0)return starts.find(v=>v>current)??null;
  let found=null;for(const v of starts)if(v<current)found=v;
  return found;
 }
 /* Visible window of the timeline in frames. */
 function zoomView(view,total,factor,anchor=.5){
  const span=clamp((view.end-view.start)/factor,Math.min(total,40),total),pivot=view.start+(view.end-view.start)*anchor;
  let start=pivot-span*anchor;start=clamp(start,0,total-span);
  return {start,end:start+span};
 }
 function panView(view,total,frames){
  const span=view.end-view.start,start=clamp(view.start+frames,0,total-span);
  return {start,end:start+span};
 }
 function followView(view,total,index){
  if(index>=view.start&&index<view.end)return view;
  const span=view.end-view.start,start=clamp(index-span/2,0,total-span);
  return {start,end:start+span};
 }
 const api={LANES,reversal,axisReversal,buildLanes,paint,nearest,runStarts,step,zoomView,panView,followView};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.Timeline=api;
})(typeof window!=='undefined'?window:this);
