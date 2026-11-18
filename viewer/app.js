'use strict';
const $=id=>document.getElementById(id);
const canvas=$('view'),ctx=canvas.getContext('2d'),gpu=document.createElement('canvas');
const gl=gpu.getContext('webgl',{antialias:true,preserveDrawingBuffer:true});
let manifest=null,frame=null,frameIndex=0,desiredIndex=0,playing=false,lastTick=0,playTime=0,generation=0;
let drag=null,cache=new Map(),inflight=new Map(),recorder=null,chunks=[];
const flightCamera=new FreeCamera(),flightKeys=new Set(),noKeys=new Set();
let lastFlightTick=null;
// The filter only thins the cloud for drawing; the algorithm always gets the full cloud.
let cloudMode='full',pointArray=null,pointColors=null,pointsAreFull=true,cloudExtent=60,overlay=null;
let modelFrames=null,modelSummary=null;
let softFrames=new Map(),softAddedCount=0,softAvailable=false;
let smoothedGeometry=new Map();
function displayGeometry(f){return $('smoothPath').checked?(smoothedGeometry.get(f?.frame)||f):f}
// Model frames can omit fields (an unknown frame has no path, points, range...).
const isNum=v=>typeof v==='number'&&Number.isFinite(v);
const fixed=(v,digits=1)=>isNum(v)?v.toFixed(digits):'—';
const count=v=>isNum(v)?v.toLocaleString('ru-RU'):'—';
const candidatesOf=f=>Array.isArray(f?.candidates)?f.candidates.filter(a=>Array.isArray(a?.bbox)&&a.bbox.length===2):[];
function modelFrame(i){const f=ModelView.frameFor(manifest.frames[i],modelFrames?.get(manifest.frames[i].frame));return $('softConfirmation').checked&&softAvailable?SoftConfirmation.apply(f,softFrames.get(f.frame)):f}
function updateSoftHint(){
 $('softHint').textContent=$('softConfirmation').checked&&softAvailable?
 `Эксперимент включён · +${softAddedCount} тревожных кадров за проезд. Габарит прежний; наружные точки пока не учитываются. Время расчёта — базового ядра.`:
 softAvailable?'Базовый детектор. Переключатель добавляет сдержанный накопитель подтверждений.':'Базовый детектор. Для текущего расчёта эксперимент не подготовлен.';
}
function trainSpeedKmh(f,i){
 // path_info.odometry_ds is metres travelled since the previous frame (~10 Hz).
 const ds=f?.path_info?.odometry_ds;if(!isNum(ds))return null;
 const prev=manifest?.frames[i-1],dt=prev?f.time_s-prev.time_s:NaN;
 return ds/(dt>.05&&dt<.3?dt:.1)*3.6;
}
const speedText=v=>v==null?'—':v.toFixed(v<10?1:0);
function fail(e,source=''){$('error').hidden=false;$('error').dataset.source=source;$('error').textContent=String(e?.message||e);console.error(e)}
const frameFail=e=>fail(e,'frame');
if(!gl)fail('WebGL недоступен. Включите аппаратное ускорение браузера и обновите страницу.');
const vertex=`attribute vec3 aPosition;attribute vec3 aColor;uniform mat4 matrix;uniform float size;varying vec3 color;void main(){gl_Position=matrix*vec4(-aPosition.y,aPosition.z,-aPosition.x,1.);gl_PointSize=size;color=aColor;}`;
const fragment=`precision mediump float;varying vec3 color;void main(){gl_FragColor=vec4(color,1.);}`;
function shader(type,source){let s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
let program,matLoc,sizeLoc,posLoc,colLoc;
const buffers={},uploaded={points:null,colors:null,overlay:null};
if(gl){program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error('WebGL link failed');gl.useProgram(program);matLoc=gl.getUniformLocation(program,'matrix');sizeLoc=gl.getUniformLocation(program,'size');posLoc=gl.getAttribLocation(program,'aPosition');colLoc=gl.getAttribLocation(program,'aColor');for(const k of ['points','colors','lines','lineColors'])buffers[k]=gl.createBuffer();gl.enable(gl.DEPTH_TEST)}
function multiply(a,b){let r=new Float32Array(16);for(let c=0;c<4;c++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)r[c*4+row]+=a[k*4+row]*b[c*4+k];return r}
function matrix(){
 const view=flightCamera.viewMatrix(),f=1/Math.tan(flightCamera.fov*Math.PI/360),near=.05;
 const far=Math.max(400,cloudExtent*4,Math.hypot(...flightCamera.position)+cloudExtent*2);
 const projection=[f/(gpu.width/gpu.height),0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0];
 return multiply(projection,view);
}
function updateCameraUI(){
 $('cameraStatus').textContent=`Свободная камера · FOV ${Math.round(flightCamera.fov)}°`;
 canvas.dataset.cameraPosition=flightCamera.position.join(',');
 canvas.dataset.cameraFov=String(flightCamera.fov);
 document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===flightCamera.mode));
}
// Point and overlay geometry are uploaded once per frame; camera motion only redraws.
function upload(buffer,data){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW)}
function draw(position,color,n,mode,size){gl.bindBuffer(gl.ARRAY_BUFFER,position);gl.enableVertexAttribArray(posLoc);gl.vertexAttribPointer(posLoc,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,color);gl.enableVertexAttribArray(colLoc);gl.vertexAttribPointer(colLoc,3,gl.FLOAT,false,0,0);gl.uniform1f(sizeLoc,size);gl.drawArrays(mode,0,n)}
function project(s,u,z,m){let v=[-u,z,-s,1],o=[0,0,0,0];for(let row=0;row<4;row++)for(let k=0;k<4;k++)o[row]+=m[k*4+row]*v[k];return o[3]>.1?[(o[0]/o[3]+1)*canvas.width/2,(1-o[1]/o[3])*canvas.height/2]:null}
function clock(t){t=isNum(t)?t:0;return `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`}
function buildOverlay(){
 const geometry=displayGeometry(frame);
 const vertices=[],colors=[];const line=(a,b,c)=>{vertices.push(...a,...b);colors.push(...c,...c)};
 if(frame&&Array.isArray(frame.floor)){const verified=isNum(frame.analysis_range_m)&&frame.analysis_range_m>0?frame.analysis_range_m:60,limit=isNum(frame.envelope_range_m)&&frame.envelope_range_m>verified?frame.envelope_range_m:verified,point=(s,u,h)=>ModelView.pathPoint(geometry,s,u,h);
  // beyond the verified range the envelope is only kept steady from the previous frames: drawn dimmer
  if($('envelope').checked){for(let s=1;s<limit;s+=1){const a=ModelView.envelopeOutline(frame,s),end=Math.min(limit,s+1),b=ModelView.envelopeOutline(frame,end),col=s<verified?[.85,.4,1]:[.5,.42,.62];a.forEach(([u,z],i)=>line(point(s,u,z),point(end,...b[i]),col))}for(let s=10;s<=limit;s+=10){const outline=ModelView.envelopeOutline(frame,s);outline.forEach(([u,z],i)=>line(point(s,u,z),point(s,...outline[(i+1)%outline.length]),[.67,.32,.9]))}}
  if($('pathLine').checked){const end=ModelView.geometryLength(frame),rawPoint=(s,u,h)=>ModelView.pathPoint(frame,s,u,h);for(let s=0;s<end;s+=.5){if(s>limit&&Math.floor(s)%3===2)continue;line(rawPoint(s,0,.4),rawPoint(Math.min(end,s+.5),0,.4),s<=limit?[.36,1,.83]:[.27,.62,.73])}line(point(limit,-1.8,.4),point(limit,1.8,.4),[.85,.93,1])}
  if($('boxes').checked){for(let a of candidatesOf(frame)){let [lo,hi]=a.bbox,c=a.confirmed?[1,.34,.39]:a.attention_strong?[1,.52,.12]:DetectorEvaluation.isAttention(a)?[1,.71,.33]:[.35,.43,.5];let corners=[];for(let i=0;i<8;i++)corners.push([i&1?hi[0]:lo[0],i&2?hi[1]:lo[1],i&4?hi[2]:lo[2]]);for(let i=0;i<8;i++)for(let j of [1,2,4])if(!(i&j))line(corners[i],corners[i|j],c)}}
 }
 Review.overlayLines(line);
 return {vertices:new Float32Array(vertices),colors:new Float32Array(colors),n:vertices.length/3};
}
// Distance colours: light cyan near the train, blue at 60 m, violet far away (red and amber stay for obstacles).
const RAMP=[[.59,.93,1],[.23,.54,1],[.5,.33,.89]];
function distanceColor(s,z){const t=Math.min(1,Math.max(0,s/(manifest?.summary?.color_range_m||150))),k=t<.4?t/.4:(t-.4)/.6,a=RAMP[t<.4?0:1],b=RAMP[t<.4?1:2],shade=.72+.28*Math.min(1,Math.max(0,(z+1.6)/5.5));return a.map((x,i)=>(x+(b[i]-x)*k)*shade)}
function pointColorsFor(points){
 const pc=new Float32Array(points.length),floor=Array.isArray(frame?.floor)?frame.floor:null,boxes=$('boxes').checked?candidatesOf(frame):[],byDistance=$('colorDistance').checked;
 for(let i=0;i<points.length;i+=3){let s=points[i],u=points[i+1],z=points[i+2],brightness=byDistance?1:Math.max(.32,1-s/180);let col=byDistance?distanceColor(s,z):[.35,.70,.72];if(floor){let height=z-(floor[0]*s+floor[1]*u+floor[2]);if(!byDistance&&height<.35)col=[.30,.41,.50];if(!byDistance&&height>2.8)col=[.29,.50,.57];for(let a of boxes){let [l,h]=a.bbox;if(s>=l[0]-.01&&s<=h[0]+.01&&u>=l[1]-.01&&u<=h[1]+.01&&z>=l[2]-.01&&z<=h[2]+.01){col=a.confirmed?[1,.4,.43]:DetectorEvaluation.isAttention(a)?[1,.75,.36]:[.35,.43,.5];brightness=1;break}}}pc[i]=col[0]*brightness;pc[i+1]=col[1]*brightness;pc[i+2]=col[2]*brightness}
 return pc;
}
function invalidateOverlay(){overlay=null;render()}
function render(){
 updateCameraUI();if(!gl)return;const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=gpu.width=w;canvas.height=gpu.height=h}gl.viewport(0,0,w,h);gl.clearColor(.027,.059,.083,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);let m=matrix();gl.uniformMatrix4fv(matLoc,false,m);
 if(pointArray?.length){if(!pointColors)pointColors=pointColorsFor(pointArray);if(uploaded.points!==pointArray){upload(buffers.points,pointArray);uploaded.points=pointArray}if(uploaded.colors!==pointColors){upload(buffers.colors,pointColors);uploaded.colors=pointColors}draw(buffers.points,buffers.colors,pointArray.length/3,gl.POINTS,1.5*dpr)}
 if(!overlay){try{overlay=buildOverlay()}catch(e){console.error('Геометрия кадра:',e);overlay={n:0}}}
 if(overlay.n){if(uploaded.overlay!==overlay){upload(buffers.lines,overlay.vertices);upload(buffers.lineColors,overlay.colors);uploaded.overlay=overlay}draw(buffers.lines,buffers.lineColors,overlay.n,gl.LINES,1)}
 ctx.drawImage(gpu,0,0);ctx.font=`${11*dpr}px 'Segoe UI',sans-serif`;ctx.fillStyle='#8ea7b6';
 try{if(Array.isArray(frame?.floor)){let lastLabel=null;for(let s of [10,20,30,40,50,60,80,100,120,150,200,250,300]){if(s>ModelView.geometryLength(frame))continue;let pos=ModelView.pathPoint(frame,s,-1.8,0),p=project(...pos,m);if(p&&p[0]>20&&p[0]<w-55&&p[1]>50&&p[1]<h-65&&(!lastLabel||Math.hypot(p[0]-lastLabel[0],p[1]-lastLabel[1])>42*dpr)){ctx.fillText(`${s} м`,p[0],p[1]);lastLabel=p}}const boxes=candidatesOf(frame);let a=boxes.find(x=>x.confirmed)||boxes[0];if(a&&$('boxes').checked&&isNum(a.distance_m)){let p=project(isNum(a.s)?a.s:a.distance_m,a.center?.[1]??0,a.bbox[1][2]+.35,m);if(p&&p[0]>0&&p[0]<w-100&&p[1]>50&&p[1]<h-60){ctx.fillStyle=a.confirmed?'#ff8a8e':'#f4bc73';ctx.fillText(`${a.distance_m.toFixed(1)} м`,p[0],p[1])}}}}catch(e){console.error('Подписи кадра:',e)}
 if(frame){const speed=trainSpeedKmh(frame,frameIndex);ctx.fillStyle='#0c1923e8';ctx.fillRect(18*dpr,h-98*dpr,Math.min(w-36*dpr,440*dpr),53*dpr);ctx.fillStyle=frame.obstacle?'#ff8a8e':frame.status==='candidate'?'#f4bc73':'#9ce4d6';ctx.font=`${14*dpr}px 'Segoe UI',sans-serif`;ctx.fillText(frame.obstacle?`ПРЕПЯТСТВИЕ   ${fixed(frame.distance_m)} м`:frame.status==='candidate'?'ВНИМАНИЕ · ожидает подтверждения':frame.status==='skipped'?'КАДР ПРОПУЩЕН ПО НАСТРОЙКЕ ГЦ':frame.status==='unknown'?'НЕДОСТАТОЧНО ДАННЫХ':'ПРЕПЯТСТВИЕ НЕ ОБНАРУЖЕНО',30*dpr,h-76*dpr);ctx.font=`${10*dpr}px 'Segoe UI',sans-serif`;ctx.fillStyle='#a6b6c4';ctx.fillText(`${clock(frame.time_s)}  /  CPU ${fixed(frame.latency_ms)} мс  /  ${speed==null?'скорость —':speedText(speed)+' км/ч'}`,30*dpr,h-58*dpr)}
}
function updatePanels(){
 canvas.dataset.frameId=String(frame.frame);canvas.dataset.analysisRange=String(frame.analysis_range_m||0);
 const info=frame.path_info,speed=trainSpeedKmh(frame,frameIndex);
 $('pathInfo').textContent=info?`Рельсы видны до ${fixed(info.rail_support_m||0,0)} м, профиль тоннеля до ${fixed(info.support_m||0,0)} м${info.coasting_frames?`, ось держится без рельсов ${info.coasting_frames} кадр.`:''}.`:frame.geometry_reason||'Ось пути в этом кадре не восстановлена';
 $('analysisRange').textContent=Array.isArray(frame.floor)?`${fixed(isNum(frame.analysis_range_m)?frame.analysis_range_m:0,0)} м`:'—';$('envelopeRange').textContent=Array.isArray(frame.floor)?`${fixed(Math.max(isNum(frame.envelope_range_m)?frame.envelope_range_m:0,isNum(frame.analysis_range_m)?frame.analysis_range_m:0),0)} м`:'—';
 $('summary').textContent=`${manifest.summary.title||manifest.summary.id} · ${modelSummary?.frames??manifest.frames.length} кадров · расчёт p95 ${fixed(modelSummary?.latency_p95_ms)} мс${$('softConfirmation').checked&&softAvailable?' · эксперимент: непрерывное подтверждение':''}`;
 $('state').textContent=frame.obstacle?'Препятствие':frame.status==='candidate'&&frame.attention_level>=2?'Внимание (усиленное)':({candidate:'Внимание',unknown:'Недостаточно данных',clear:'Путь свободен'}[frame.status]||'Нет решения');
 $('decision').className='decision '+(frame.obstacle?'danger':frame.status==='candidate'?'pending':frame.status==='unknown'?'unknown':'');
 $('distance').textContent=fixed(frame.distance_m);
 $('decisionNote').textContent=frame.soft_confirmation?'Эксперимент: добавлено подтверждение по накоплению свидетельств':frame.obstacle?'До ближайшей точки препятствия вдоль пути':frame.status==='candidate'?(frame.attention_level>=2?'Один и тот же объект долго рядом с границей габарита; подтверждения нет, но внимание усилено':'Объект в габарите или рядом с границей; подтверждения нет'):frame.status==='unknown'?(frame.geometry_reason||'Ось пути не восстановлена'):'В пределах проверенной дальности';
 $('trainSpeed').textContent=speed==null?'—':`${speedText(speed)} км/ч`;
 $('latency').textContent=isNum(frame.latency_ms)?frame.latency_ms.toFixed(0)+' мс':'—';
 $('rawPoints').textContent=count(frame.points);
 updateCloudText();
 $('clock').textContent=`${clock(frame.time_s)} / ${clock(manifest.summary.duration_s)}`;$('seek').value=frameIndex;
}
function updateCloudText(){
 const shown=pointArray?pointArray.length/3:0;
 $('pointLabel').textContent=`${count(shown)} точек · ${cloudMode==='full'&&pointsAreFull?'все точки кадра':'облегчённый показ'}`;
}
function updateUI(){
 if(!frame)return;
 Review.frameChanged();
 // A missing field must never block drawing the frame itself.
 try{updatePanels()}catch(e){console.error('Панель кадра:',e)}
 render();
}
function getPoints(i){
 const key=`${cloudMode}:${i}`;
 if(cache.has(key)){const hit=cache.get(key);cache.delete(key);cache.set(key,hit);return Promise.resolve(hit)}
 if(inflight.has(key))return inflight.get(key);
 const token=generation,mode=cloudMode,record=manifest.frames[i];
 const full=mode==='full'&&!!record.full_file,file=full?record.full_file:record.file;
 const job=(async()=>{
  if(!file)throw Error('Для кадра нет облака точек.');
  const response=await fetch(file);if(!response.ok)throw Error('Не удалось загрузить '+file);
  const entry={points:new Float32Array(await response.arrayBuffer()),full};
  if(token===generation&&mode===cloudMode){cache.set(key,entry);while(cache.size>(mode==='full'?20:48))cache.delete(cache.keys().next().value)}
  return entry;
 })();
 inflight.set(key,job);
 const done=()=>{if(inflight.get(key)===job)inflight.delete(key)};job.then(done,done);
 return job;
}
function prefetch(i){
 // Keep the next few frames on hand so playback does not wait on the network.
 const hz=+$('displayHz').value,step=hz?Math.max(1,Math.round((manifest.summary.native_hz||10)/hz)):1;
 const ahead=playing?(cloudMode==='full'?8:16):3;
 for(let k=1;k<=ahead;k++){const j=i+k*step;if(j>=manifest.frames.length)break;getPoints(j).catch(()=>{})}
}
async function showFrame(i){
 if(!manifest)return;
 i=Math.max(0,Math.min(manifest.frames.length-1,i));desiredIndex=i;
 const token=generation,mode=cloudMode,entry=await getPoints(i);
 if(token!==generation||mode!==cloudMode||i!==desiredIndex)return;
 pointArray=entry.points;pointsAreFull=entry.full;pointColors=null;overlay=null;frame=modelFrame(i);frameIndex=i;
 if(playing&&!+$('displayHz').value)playTime=Math.min(playTime,(frame.time_s||0)+.05);
 const bounds=frame.full_bounds;if(Array.isArray(bounds)&&bounds.length===2)cloudExtent=Math.hypot(...bounds[1].map((x,k)=>x-bounds[0][k]))||cloudExtent;
 if(!$('error').hidden&&$('error').dataset.source==='frame')$('error').hidden=true;
 updateUI();prefetch(i);
}
function setPlaying(value){playing=value;lastTick=performance.now();playTime=frame?.time_s||0;$('play').textContent=value?'Ⅱ Пауза':'▶ Воспроизвести';if(value&&manifest)prefetch(frameIndex)}
async function fetchModel(){
 const spec=manifest.models?.find(m=>m.id==='tunnel');
 if(!spec?.file)throw Error('Для проезда нет результатов. Выполните scripts/replay_bag.py.');
 const response=await fetch(spec.file,{cache:'no-store'});if(!response.ok)throw Error('Не удалось загрузить результаты');
 const bytes=await response.arrayBuffer(),payload=JSON.parse(new TextDecoder().decode(bytes)),byId=new Map(payload.frames.map(f=>[f.frame,f]));
 softFrames=new Map();softAddedCount=0;softAvailable=false;
 const experiment=manifest.models?.find(m=>m.id==='soft_confirmation');
 if(experiment){
  try{
   const r=await fetch(experiment.file,{cache:'no-store'});if(!r.ok)throw Error('Experiment unavailable');
   const data=await r.json(),digest=await crypto.subtle.digest('SHA-256',bytes);
   const hash=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
   if(hash===data.source_sha256){softFrames=new Map(data.frames.map(f=>[f.frame,f.candidate_indices]));softAddedCount=data.added_alarm_frames;softAvailable=true}
  }catch(e){console.warn('Эксперимент недоступен:',e)}
 }
 $('softConfirmation').disabled=!softAvailable;updateSoftHint();
 smoothedGeometry=PathSmoothing.build(payload.frames);
 return {frames:byId,summary:payload.summary};
}
function lockSelectors(locked){
 for(const id of ['sequence','cloudMode','play','prev','next','seek'])$(id).disabled=locked;
 $('play').disabled=locked||(manifest?.frames.length||0)<2;
 $('softConfirmation').disabled=locked||!softAvailable;
}
async function loadSequence(id){
 setPlaying(false);generation++;cache.clear();inflight.clear();frame=null;pointArray=null;pointColors=null;overlay=null;lockSelectors(true);$('error').hidden=true;
 const response=await fetch(`data/${id}/manifest.json`,{cache:'no-store'});if(!response.ok)throw Error('Не найден manifest');manifest=await response.json();
 const hasFullCloud=manifest.frames.some(f=>f.full_file);
 $('cloudMode').querySelector('option[value="full"]').disabled=!hasFullCloud;
 if(!hasFullCloud){cloudMode='filtered';$('cloudMode').value=cloudMode}
 flightKeys.clear();flightCamera.teleport('cab');
 const loaded=await fetchModel();modelFrames=loaded.frames;modelSummary=loaded.summary;
 await Review.load(id,loaded);
 $('sceneTitle').textContent=manifest.summary.title||id;$('sceneHint').textContent=manifest.summary.hint||'';$('seek').max=manifest.frames.length-1;
 await showFrame(0);lockSelectors(false);
}
$('smoothPath').onchange=invalidateOverlay;
$('softConfirmation').onchange=()=>{if(!manifest||!frame)return;frame=modelFrame(frameIndex);pointColors=null;overlay=null;updateSoftHint();updateUI();Review.refresh()};
$('play').onclick=()=>{if(!playing&&frameIndex===manifest.frames.length-1){showFrame(0).then(()=>setPlaying(true)).catch(frameFail)}else setPlaying(!playing)};
$('prev').onclick=()=>{setPlaying(false);showFrame(frameIndex-1).catch(frameFail)};$('next').onclick=()=>{setPlaying(false);showFrame(frameIndex+1).catch(frameFail)};$('seek').oninput=()=>{setPlaying(false);showFrame(+$('seek').value).catch(frameFail)};$('sequence').onchange=()=>loadSequence($('sequence').value).catch(e=>{fail(e);lockSelectors(false)});
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>{
 flightKeys.clear();flightCamera.teleport(b.dataset.camera);canvas.focus({preventScroll:true});render();
});
for(const id of ['envelope','pathLine'])$(id).onchange=invalidateOverlay;
$('colorDistance').onchange=()=>{pointColors=null;render()};
$('boxes').onchange=()=>{pointColors=null;invalidateOverlay()};
canvas.onpointerdown=e=>{
 if(e.button!==0&&e.button!==2)return;
 e.preventDefault();canvas.focus({preventScroll:true});drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);
};
// Mouse turns are queued and eased in by the animation loop, one redraw per display frame.
canvas.onpointermove=e=>{
 if(!drag)return;
 flightCamera.turn(e.clientX-drag[0],e.clientY-drag[1]);drag=[e.clientX,e.clientY];
};
const stopDrag=()=>drag=null;
canvas.onpointerup=stopDrag;canvas.onpointercancel=stopDrag;canvas.onlostpointercapture=stopDrag;
canvas.oncontextmenu=e=>e.preventDefault();
canvas.addEventListener('wheel',e=>{
 e.preventDefault();const unit=e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1;
 flightCamera.zoom(e.deltaY*unit);
},{passive:false});
const stopFlight=()=>{flightKeys.clear();drag=null};
window.addEventListener('blur',stopFlight);
canvas.addEventListener('blur',stopFlight);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopFlight();lastFlightTick=null}});
$('cloudMode').onchange=async()=>{
 if(!manifest)return;const previous=cloudMode;setPlaying(false);generation++;cache.clear();inflight.clear();
 cloudMode=$('cloudMode').value;lockSelectors(true);$('error').hidden=true;
 $('pointLabel').textContent='Загрузка облака…';
 try{await showFrame(frameIndex)}catch(e){cloudMode=previous;$('cloudMode').value=previous;updateUI();fail(e)}
 finally{lockSelectors(false)}
};
$('fitCloud').onclick=()=>{
 if(!pointArray?.length)return;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let i=0;i<pointArray.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],pointArray[i+k]);hi[k]=Math.max(hi[k],pointArray[i+k])}
 flightKeys.clear();flightCamera.fitBounds([-hi[1],lo[2],-hi[0]],[-lo[1],hi[2],-lo[0]],canvas.clientWidth/canvas.clientHeight);
 canvas.focus({preventScroll:true});render();
};
const flightCodes=new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight']);
document.addEventListener('keydown',e=>{
 if(e.target.closest('input,select,textarea,button,[contenteditable="true"]'))return;
 if(e.code==='Escape'){stopFlight();canvas.blur();return}
 if(document.activeElement===canvas&&flightCodes.has(e.code)){
  if(e.ctrlKey||e.metaKey||e.altKey)return;
  e.preventDefault();flightKeys.add(e.code);return;
 }
 if(e.code==='Space'&&!e.repeat){e.preventDefault();$('play').click()}
});
document.addEventListener('keyup',e=>flightKeys.delete(e.code));
$('record').onclick=()=>{if(recorder?.state==='recording'){recorder.stop();return}if(!canvas.captureStream||!window.MediaRecorder){fail('Запись не поддерживается этим браузером.');return}chunks=[];const options=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?{mimeType:'video/webm;codecs=vp9'}:{};recorder=new MediaRecorder(canvas.captureStream(30),options);recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob(chunks,{type:recorder.mimeType}));a.href=url;a.download=`metro-${$('sequence').value}.webm`;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);$('record').textContent='● Записать WebM'};recorder.start();$('record').textContent='■ Сохранить WebM';if(!playing)setPlaying(true)};
function advancePlayback(now){
 let delta=(now-lastTick)/1000;playTime+=Math.min(delta,.2)*+$('speed').value;let i=frameIndex;const hz=+$('displayHz').value;
 if(!hz){if(i+1<manifest.frames.length&&manifest.frames[i+1].time_s<=playTime)i++}
 else{while(i+1<manifest.frames.length&&manifest.frames[i+1].time_s<=playTime)i++;if(i>frameIndex&&i<manifest.frames.length-1&&manifest.frames[i].time_s-(frame?.time_s||0)<1/hz)i=frameIndex;}
 if(i!==desiredIndex)showFrame(i).catch(e=>{setPlaying(false);frameFail(e)});
 if(playTime>=manifest.summary.duration_s&&frameIndex===manifest.frames.length-1){setPlaying(false);if(recorder?.state==='recording')recorder.stop()}
}
function tick(now){
 const dt=lastFlightTick===null?0:(now-lastFlightTick)/1000;lastFlightTick=now;
 const cameraMoved=flightCamera.update(document.activeElement===canvas?flightKeys:noKeys,dt);
 if(playing&&manifest)advancePlayback(now);
 lastTick=now;if(cameraMoved||recorder?.state==='recording')render();requestAnimationFrame(tick);
}
new ResizeObserver(()=>{render();if(manifest)Review.redraw()}).observe(canvas);
Review.init();
(async()=>{try{
 const r=await fetch('data/index.json',{cache:'no-store'});if(!r.ok)throw Error('Нет подготовленных данных. Выполните python scripts/make_demo.py или импортируйте свою запись через scripts/replay_bag.py.');
 // Dataset previews (no detector output) are not recordings to review here.
 const sequences=(await r.json()).filter(s=>s?.id&&s.kind!=='dataset_preview');
 if(!sequences.length)throw Error('Нет подготовленных данных. Выполните python scripts/make_demo.py или импортируйте свою запись через scripts/replay_bag.py.');
 sequences.sort((a,b)=>(a.title||a.id).localeCompare(b.title||b.id));
 for(let s of sequences){let o=document.createElement('option');o.value=s.id;o.textContent=s.title||s.id;$('sequence').append(o)}
 // ?seq=<id>&frame=<n>&camera=cab opens a given moment, handy for sharing and screenshots
 const q=new URLSearchParams(location.search),wanted=sequences.find(x=>x.id===q.get('seq'))?.id||sequences[0].id;
 $('softConfirmation').checked=q.get('soft')==='1';
 $('sequence').value=wanted;requestAnimationFrame(tick);await loadSequence(wanted);
 if(q.has('frame'))await showFrame(+q.get('frame'));
 if(q.has('camera')){flightCamera.teleport(q.get('camera'));render()}
}catch(e){fail(e);lockSelectors(false)}})();
