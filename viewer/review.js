/* Review of a saved result by eye: statistics under the scene and a multi-level timeline whose
   FP and FN lanes open the frame in the scene. There are no detector parameters to turn here. */
const Review=(()=>{
 const POLICY={pad:.2,ambiguous:'ignore',maxRange:300};   // fixed: every result is judged the same way
 const NEW_AXIS=' · без разворота оси';
 let truth=null,truthFrames=new Map(),baseline=null,baselineSummary=null,runs=[],selected='recorded',evaluation=null,lanes=null,focus=null,loadToken=0;
 let view={start:0,end:1},indexById=new Map(),laneDom=new Map(),pointer=null;
 const text=(tag,content,className)=>{const el=document.createElement(tag);el.textContent=content;if(className)el.className=className;return el};
 const errorName=type=>({FP:'Ложная тревога',FN:'Пропуск',DUP:'Вторая рамка на найденной цели',SHIFT:'Рамка сдвинута относительно цели',WFN:'Нет внимания к спорному кубу',WFP:'Лишнее внимание'})[type]||type;
 const pct=(a,b)=>b?`${(100*a/b).toFixed(a&&a/b<.1?1:0)}%`:'—';
 const dpr=()=>Math.min(devicePixelRatio||1,2);

 // ---------------------------------------------------------------- loading
 async function load(sequence,loaded){
  const token=++loadToken;baseline=loaded.frames;baselineSummary=loaded.summary;truth=null;truthFrames=new Map();focus=null;evaluation=null;lanes=null;
  $('labRun').disabled=true;
  const [gt,runResponse]=await Promise.all([
   fetch(`data/${sequence}/ground-truth.json`,{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null),
   fetch(`/api/experiments?sequence=${encodeURIComponent(sequence)}`).then(r=>r.ok?r.json():{runs:[]}).catch(()=>({runs:[]}))
  ]);
  if(token!==loadToken)return;
  truth=gt;runs=runResponse.runs||[];
  if(gt)truthFrames=new Map(gt.frames.map(f=>[f.frame,f]));
  const ambiguous=gt?.frames.some(f=>f.objects.some(o=>o.label==='ambiguous'));
  $('truthStatus').textContent=gt?`${gt.source||'Локальная разметка'} · ${gt.frames.length} синхронных кадров. ${gt.assumed_clean?'Рабочее предположение о чистом проезде; эталонных рамок здесь нет.':ambiguous?'Спорные кубы оцениваются отдельно.':'Явно наружные объекты остаются отрицательными.'}`:'Для этого проезда нет покадровой разметки: FP и FN не считаются.';
  indexById=new Map(manifest.frames.map((f,i)=>[f.frame,i]));
  fillRunSelect();
  const requested=new URLSearchParams(location.search).get('run');
  const choice=[...$('labRun').options].some(o=>o.value===requested)?requested:preferredRun();
  await activate(choice,false);$('labRun').disabled=false;
 }
 /* One entry per kind of result (the newest), so the list stays short. */
 function fillRunSelect(){
  const select=$('labRun');select.replaceChildren();
  const add=(value,label)=>{const o=text('option',label);o.value=value;select.append(o)};
  add('recorded','Базовый детектор');if(softAvailable)add('soft-recorded','Сдержанный CUSUM-like · сохранённый');
  const seen=new Set();
  for(const r of runs){if(seen.has(r.label))continue;seen.add(r.label);add(r.id,r.label)}
 }
 // the core's own answer is what the ROS node publishes: show it first, then the older experiments
 const preferredRun=()=>runs.find(r=>r.version==='core')?.id||runs.find(r=>r.version==='soft-spatial'&&r.label.endsWith(NEW_AXIS))?.id||'recorded';

 async function activate(id,redraw=true){
  setPlaying(false);selected=id;focus=null;
  if(id==='recorded'||id==='soft-recorded'){
   modelFrames=baseline;modelSummary=baselineSummary;$('softConfirmation').checked=id==='soft-recorded';
  }else{
   const meta=runs.find(r=>r.id===id);if(!meta)throw Error('Вариант не найден');
   const response=await fetch(meta.file,{cache:'no-store'});if(!response.ok)throw Error('Результат не загружен');
   const payload=await response.json();
   if(payload.frames.length!==manifest.frames.length||payload.frames.some((f,i)=>f.frame!==manifest.frames[i].frame||Math.abs(f.time_s-manifest.frames[i].time_s)>.001))throw Error('Этот вариант рассчитан для другой последовательности кадров');
   modelFrames=new Map(payload.frames.map(f=>[f.frame,f]));modelSummary=payload.summary;$('softConfirmation').checked=false;
  }
  $('labRun').value=id;smoothedGeometry=PathSmoothing.build([...modelFrames.values()]);
  updateSoftHint();
  if(!['recorded','soft-recorded'].includes(id))$('softHint').textContent=`${runs.find(r=>r.id===id)?.label||modelSummary.detector_version}. Облака и разметка общие.${modelSummary.detector_version?.startsWith('soft')?' Время кадра — базового C++.':''}`;
  const url=new URL(location.href);url.searchParams.set('seq',manifest.summary.id);url.searchParams.set('run',id);url.searchParams.set('soft',id==='soft-recorded'?'1':'0');history.replaceState(null,'',url);
  if(redraw&&frame){frame=modelFrame(frameIndex);pointColors=null;overlay=null;updateUI()}
  refresh();
 }

 // ---------------------------------------------------------------- statistics
 function card(label,value,note,className=''){
  const el=text('div','',`metric-card ${className}`);el.append(text('span',label),text('b',String(value)));if(note)el.append(text('small',note));return el;
 }
 function refresh(){
  const current=manifest.frames.map((_,i)=>modelFrame(i));
  evaluation=truth?DetectorEvaluation.evaluate(current,truth,POLICY):null;
  lanes=Timeline.buildLanes({frames:current,truth,evaluation});
  view={start:0,end:Math.max(1,current.length)};
  renderStats(current);buildLaneDom();drawLanes();playheadAt(frameIndex);
  $('errorFocus').textContent='';
 }
 function renderStats(current){
  const algo=$('statsAlgo'),truthBox=$('statsTruth'),objects=$('objectMetrics');
  algo.replaceChildren();truthBox.replaceChildren();objects.replaceChildren();
  const n=current.length,by=f=>f.obstacle?'obstacle':f.status==='candidate'?'candidate':f.status==='unknown'||f.status==='skipped'?'unknown':'clear';
  const tally={obstacle:0,candidate:0,unknown:0,clear:0};for(const f of current)tally[by(f)]++;
  const latency=current.map(f=>f.latency_ms).filter(Number.isFinite).sort((a,b)=>a-b),q=p=>latency.length?latency[Math.min(latency.length-1,Math.floor(p*latency.length))]:null;
  const alarms=Timeline.runStarts(lanes.indices.decision.filter(i=>lanes.cells.decision[i]===4)).length;
  const axisInside=lanes.cells.axis.reduce((a,c)=>a+(c===2),0),axisBeyond=lanes.cells.axis.reduce((a,c)=>a+(c===1),0);
  algo.append(
   card('Кадров',count(n),`${clock(manifest.summary.duration_s)} записи`),
   card('Препятствие',count(tally.obstacle),`${pct(tally.obstacle,n)} кадров · эпизодов ${alarms}`,'metric-fp-soft'),
   card('Внимание',count(tally.candidate),pct(tally.candidate,n)+' кадров'),
   card('Нет решения',count(tally.unknown),pct(tally.unknown,n)+' кадров'),
   card('Расчёт кадра',latency.length?`${q(.95).toFixed(0)} мс`:'—',latency.length?`p95 · медиана ${q(.5).toFixed(0)} мс`:''),
   card('Ось изгибается обратно',count(axisInside+axisBeyond),`за концом данных ${count(axisBeyond)} · в измеренной части ${count(axisInside)}`,axisInside+axisBeyond?'metric-axis':''));
  if(!evaluation){truthBox.append(text('p','Для этого проезда нет покадровой разметки: FP и FN не считаются, проверяйте решения по сцене и дорожкам «Решение» и «Ось».','hint'));$('labExport').disabled=true;return}
  const c=evaluation.counts,b=evaluation.boundary;
  truthBox.append(
   card('TP · объект-кадры',count(c.tp),'цель в разметке и тревога на ней','metric-tp'),
   card('FP · ложные тревоги',count(c.fp),c.fp?`удержанные рамки без точек ${count(c.fpStale)} · у внешних объектов ${count(c.fpExterior)}`:'нет','metric-fp'),
   card('Рамка у цели',count(c.dup+c.shifted),c.dup+c.shifted?`вторая рамка на цели ${count(c.dup)} · сдвиг до ${DetectorEvaluation.NEAR} м ${count(c.shifted)} · в FP и FN не входят`:'нет','metric-near'),
   card('FN · пропуски',count(c.fn),c.fn?`за проверенной дальностью ${count(c.fnBeyond)} · внутри ${count(c.fn-c.fnBeyond)}`:'нет','metric-fn'),
   card('Внимание · кадры',count(c.attention),'сигнал без подтверждения, в TP не входит'),
   card('Цели найдены',`${evaluation.detected} / ${evaluation.total}`,`хотя бы в одном кадре · исключено спорных ${count(c.ignored)}`));
  if(b.counts.visible)truthBox.append(card('Спорные кубы · замечены',`${b.counts.covered} / ${b.counts.visible}`,`подтверждение ${b.counts.confirmed} · внимание ${b.counts.attention} · пропуск ${b.counts.missed} · лишнее внимание ${b.counts.falseAttention}`));
  for(const o of evaluation.objects)objects.append(text('p',`${o.name}: найден в ${o.tp} из ${o.visible} кадров${o.shifted?` · со сдвигом рамки ${o.shifted}`:''} · пропусков ${o.fn} · первая тревога ${o.firstDistance==null?'—':o.firstDistance.toFixed(1)+' м по X'}`));
  for(const o of b.objects)objects.append(text('p',`Спорный · ${o.name}: замечен ${o.covered}/${o.visible} · внимание ${o.attention} · подтверждение ${o.confirmed} · самый долгий пропуск ${o.longestMiss} кадров`));
  $('labExport').disabled=false;
 }

 // ---------------------------------------------------------------- lanes
 function buildLaneDom(){
  const box=$('laneRows');box.replaceChildren();laneDom=new Map();
  for(const lane of Timeline.LANES){
   const needsTruth=['truth','fp','fn','near','disputed'].includes(lane.id);
   if(needsTruth&&!truth)continue;
   if(lane.id==='disputed'&&!truth.frames.some(f=>f.objects.some(o=>o.label==='ambiguous')))continue;
   const row=text('div','','lane'),label=text('div','','lane-label'),canvas=document.createElement('canvas');
   row.dataset.lane=lane.id;label.title=lane.hint;
   label.append(text('span',lane.label,'lane-name'),text('b','','lane-count'));
   if(lane.step){
    const prev=text('button','‹','lane-step'),next=text('button','›','lane-step');
    prev.type='button';next.type='button';prev.setAttribute('aria-label',`${lane.label}: предыдущее событие`);next.setAttribute('aria-label',`${lane.label}: следующее событие`);
    prev.onclick=()=>gotoEvent(lane,-1);next.onclick=()=>gotoEvent(lane,1);label.append(prev,next);
   }
   canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`${lane.label}: ${lane.hint}`);canvas.tabIndex=-1;
   canvas.addEventListener('pointerdown',e=>laneDown(lane,e));
   canvas.addEventListener('pointermove',e=>laneMove(lane,e));
   canvas.addEventListener('pointerup',e=>laneUp(lane,e));
   canvas.addEventListener('pointerleave',()=>{$('laneTip').hidden=true});
   canvas.addEventListener('pointercancel',()=>{pointer=null});
   canvas.addEventListener('wheel',e=>{if(!(e.ctrlKey||e.metaKey||e.shiftKey))return;e.preventDefault();zoom(e.deltaY<0?1.4:1/1.4,anchorOf(canvas,e))},{passive:false});
   row.append(label,canvas);box.append(row);laneDom.set(lane.id,{lane,canvas,count:label.querySelector('.lane-count')});
  }
 }
 const anchorOf=(canvas,e)=>{const r=canvas.getBoundingClientRect();return Math.min(1,Math.max(0,(e.clientX-r.left)/r.width))};
 const frameAt=(canvas,e)=>view.start+anchorOf(canvas,e)*(view.end-view.start);
 function drawLanes(){
  if(!lanes)return;
  const ratio=dpr();
  for(const {lane,canvas,count} of laneDom.values()){
   const width=Math.max(1,Math.round(canvas.clientWidth*ratio)),height=Math.round(18*ratio);
   if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
   const g=canvas.getContext('2d');g.fillStyle='#142230';g.fillRect(0,0,width,height);
   const painted=Timeline.paint(lanes.cells[lane.id],view.start,view.end,width),fat=lane.step?Math.ceil(2*ratio):1;
   for(let x=0;x<width;x++){const code=painted[x];if(!code)continue;g.fillStyle=lane.codes[code][0];g.fillRect(lane.step?Math.max(0,x-fat):x,0,lane.step?2*fat+1:1,height)}
   const runsCount=Timeline.runStarts(lanes.indices[lane.id]).length;
   count.textContent=lane.step||lane.id==='attention'?String(runsCount):'';count.title=runsCount?`${runsCount} эпизодов`:'нет';
  }
  $('viewRange').textContent=view.end-view.start>=manifest.frames.length-.5?'весь проезд':`${clock(timeOf(view.start))} – ${clock(timeOf(view.end-1))}`;
  $('zoomAll').disabled=$('zoomOut').disabled=view.end-view.start>=manifest.frames.length-.5;
 }
 const timeOf=i=>manifest.frames[Math.max(0,Math.min(manifest.frames.length-1,Math.round(i)))].time_s;
 function playheadAt(index){
  if(!lanes)return;
  const followed=Timeline.followView(view,manifest.frames.length,index),head=$('playhead'),first=[...laneDom.values()][0];
  if(!first){head.hidden=true;return}
  if(followed!==view){view=followed;drawLanes()}
  const area=$('laneArea').getBoundingClientRect(),c=first.canvas.getBoundingClientRect();
  head.hidden=false;head.style.left=`${c.left-area.left+(index+.5-view.start)/(view.end-view.start)*c.width}px`;
 }
 function zoom(factor,anchor=.5){view=Timeline.zoomView(view,manifest.frames.length,factor,anchor);drawLanes();playheadAt(frameIndex)}
 function describe(lane,index){
  const f=manifest.frames[index],code=lanes.cells[lane.id][index],base=`${clock(f.time_s)} · кадр ${f.frame}`;
  if(!code)return base;
  const row=evaluation?.perFrame.get(f.frame),names=type=>[...new Set((row?.errors||[]).filter(e=>type.includes(e.type)).map(e=>e.name))].join(', ');
  const detail=lane.errors?names(lane.errors):'';
  return `${base} · ${lane.codes[code][1]}${detail?` · ${detail}`:''}`;
 }

 // ---------------------------------------------------------------- pointer handling
 function laneDown(lane,e){pointer={lane,x:e.clientX,view:{...view},moved:false};e.currentTarget.setPointerCapture(e.pointerId)}
 function laneMove(lane,e){
  const canvas=e.currentTarget,span=view.end-view.start;
  if(pointer&&e.buttons&&(pointer.moved||Math.abs(e.clientX-pointer.x)>4)&&span<manifest.frames.length-.5){
   pointer.moved=true;const shift=-(e.clientX-pointer.x)/canvas.getBoundingClientRect().width*span;
   view=Timeline.panView(pointer.view,manifest.frames.length,shift);drawLanes();playheadAt(frameIndex);return;
  }
  const index=Math.max(0,Math.min(manifest.frames.length-1,Math.round(frameAt(canvas,e)-.5))),tip=$('laneTip'),area=$('laneArea').getBoundingClientRect();
  tip.textContent=describe(lane,index);tip.hidden=false;
  tip.style.left=`${Math.min(area.width-260,Math.max(0,e.clientX-area.left+12))}px`;tip.style.top=`${canvas.getBoundingClientRect().top-area.top-30}px`;
 }
 function laneUp(lane,e){
  const state=pointer;pointer=null;if(!state||state.moved)return;
  const canvas=e.currentTarget,width=canvas.getBoundingClientRect().width,target=frameAt(canvas,e)-.5,tolerance=Math.max(1,7*(view.end-view.start)/width);
  const hit=lane.id==='truth'||lane.id==='decision'?null:Timeline.nearest(lanes.indices[lane.id],Math.round(target),tolerance);
  openIndex(lane,hit??Math.max(0,Math.min(manifest.frames.length-1,Math.round(target)))).catch(frameFail);
 }
 /* Open a frame; for an error lane also fly to the object and highlight it. */
 async function openIndex(lane,index){
  const id=manifest.frames[index].frame,row=evaluation?.perFrame.get(id);
  const wanted=lane.errors||null;
  const error=wanted&&row?.errors.filter(e=>wanted.includes(e.type)).sort((a,b)=>(a.distance??1e9)-(b.distance??1e9))[0];
  if(error)return openError(error);
  focus=null;$('errorFocus').textContent='';setPlaying(false);await showFrame(index);
 }
 function gotoEvent(lane,direction){
  const index=Timeline.step(lanes.indices[lane.id],frameIndex,direction);
  if(index===null){$('laneTip').hidden=false;$('laneTip').textContent=direction>0?'Дальше событий нет':'Раньше событий нет';return}
  openIndex(lane,index).catch(frameFail);
 }
 async function openError(error){
  setPlaying(false);focus=error;
  const index=indexById.get(error.frame);if(index===undefined)return;
  await showFrame(index);
  const bbox=error.parts[0],center=bbox[0].map((x,k)=>(x+bbox[1][k])/2),target=[-center[1],center[2],-center[0]];
  flightCamera.lookAt([target[0]+4,target[1]+3,target[2]+8],target);flightCamera.mode='free';
  overlay=null;render();
  $('errorFocus').textContent=`${errorName(error.type)} · ${clock(error.time_s)} · ${error.name}. Жёлтая рамка — выбранное событие. Можно отлететь WASD или вернуться кнопкой «Из кабины».`;
  const url=new URL(location.href);url.searchParams.set('seq',manifest.summary.id);url.searchParams.set('frame',index);history.replaceState(null,'',url);
  $('view').scrollIntoView({block:'center',behavior:'smooth'});
 }

 // ---------------------------------------------------------------- scene hooks
 function overlayLines(line){
  if(!frame)return;
  const box=(bbox,color)=>{const [lo,hi]=bbox,corners=Array.from({length:8},(_,i)=>[i&1?hi[0]:lo[0],i&2?hi[1]:lo[1],i&4?hi[2]:lo[2]]);for(let i=0;i<8;i++)for(const j of [1,2,4])if(!(i&j))line(corners[i],corners[i|j],color)};
  if($('showTruth').checked)for(const obj of truthFrames.get(frame.frame)?.objects||[])for(const b of obj.parts)box(b,obj.label==='positive'?[.25,.75,1]:obj.label==='ambiguous'?[.72,.56,1]:[.38,.43,.49]);
  if(focus?.frame===frame?.frame)for(const b of focus.parts)box([b[0].map(v=>v-.025),b[1].map(v=>v+.025)],[1,1,.25]);
 }
 /* Called on every shown frame: playhead, badge and what the markup says about this frame. */
 function frameChanged(){
  if(!frame)return;
  const visible=focus?.frame===frame.frame;$('errorBadge').hidden=!visible;if(visible)$('errorBadge').textContent=`${errorName(focus.type)} · ${focus.name} · ${fixed(focus.distance)} м`;
  playheadAt(frameIndex);
  const box=$('frameVerdict');box.replaceChildren();
  if(!truth){box.append(text('p','Нет покадровой разметки.','hint'));return}
  const gt=truthFrames.get(frame.frame);
  if(!gt){box.append(text('p','Кадр вне разметки.','hint'));return}
  const kinds={positive:'цель',ambiguous:'спорный',negative:'снаружи'},list=gt.objects.map(o=>`${o.name} (${kinds[o.label]||o.label})`);
  box.append(text('p',list.length?`В разметке: ${list.join(', ')}`:'В разметке пусто: цели нет','hint'));
  const errors=evaluation?.perFrame.get(frame.frame)?.errors||[];
  if(!errors.length)box.append(text('p','Ошибок в этом кадре нет.','hint ok'));
  for(const error of errors){
   const chip=text('button',`${({WFN:'Нет внимания',WFP:'Лишнее внимание',DUP:'Дубль рамки',SHIFT:'Сдвиг рамки'})[error.type]||error.type} · ${error.name}${Number.isFinite(error.distance)?` · ${error.distance.toFixed(0)} м`:''}${error.beyondVerified?' · за проверенной дальностью':''}${error.stale?' · удержанная рамка без точек':''}`,`error-entry error-${error.type}`);
   chip.type='button';chip.onclick=()=>openError(error).catch(frameFail);box.append(chip);
  }
 }

 function init(){
  $('labRun').onchange=()=>activate($('labRun').value).catch(frameFail);
  $('showTruth').onchange=invalidateOverlay;
  $('zoomIn').onclick=()=>zoom(2);$('zoomOut').onclick=()=>zoom(.5);$('zoomAll').onclick=()=>{view={start:0,end:manifest.frames.length};drawLanes();playheadAt(frameIndex)};
  $('labExport').onclick=()=>{
   if(!evaluation)return;
   const report={sequence:manifest.summary.id,run:selected,label:$('labRun').selectedOptions[0]?.textContent,policy:POLICY,source:truth.source,counts:evaluation.counts,boundary:evaluation.boundary.counts,objects:evaluation.objects,episodes:evaluation.episodes};
   const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`evaluation-${manifest.summary.id}-${selected}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  window.addEventListener('resize',()=>{if(lanes){drawLanes();playheadAt(frameIndex)}});
 }
 return {init,load,refresh,redraw:()=>{if(lanes){drawLanes();playheadAt(frameIndex)}},overlayLines,frameChanged};
})();
