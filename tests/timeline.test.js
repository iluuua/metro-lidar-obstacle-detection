const test=require('node:test'),assert=require('node:assert/strict');
const T=require('../viewer/timeline.js');
const {evaluate}=require('../viewer/evaluation.js');
const X=Array.from({length:166},(_,i)=>i*2);
const frame=(n,extra={})=>({frame:n,time_s:n*.1,status:'clear',obstacle:false,candidates:[],...extra});

test('axis reversal agrees with eval/axis_reversal.py and ignores one-way bends',()=>{
 assert.equal(T.reversal(X,X.map(x=>x*.01)),0);
 assert.equal(T.reversal(X,X.map(x=>x*x/600)),0);
 assert.equal(T.reversal(X,X.map(x=>-x*x/600)),0);
 const sBend=X.map(x=>x<150?x*x/600:150*150/600+(x-150)*.5-(x-150)**2/600);
 assert.ok(Math.abs(T.reversal(X,sBend)-0.9152881347801773)<1e-9);
 const hook=X.map(x=>x<100?x*x/400:25+(x-100)*.5-(x-100)**2/200);
 assert.ok(Math.abs(T.reversal(X,hook)-0.9092793121928464)<1e-9);
});
test('reversal is attributed to the measured part or to the extension',()=>{
 const y=X.map(x=>x<150?x*x/600:150*150/600+(x-150)*.5-(x-150)**2/300);
 const path=X.map((x,i)=>[x,y[i],0,x]);
 const beyond=T.axisReversal({status:'clear',path,path_info:{support_m:100}}),inside=T.axisReversal({status:'clear',path,path_info:{support_m:280}});
 assert.equal(beyond.inside,0);assert.ok(beyond.beyond>.15);assert.ok(inside.inside>.15);assert.equal(inside.beyond,0);
 assert.equal(T.axisReversal({status:'unknown',path}),null);assert.equal(T.axisReversal({status:'clear',path:[[0,0,0,0]]}),null);
});
test('lanes mark decision, truth, attention, axis and per-frame errors',()=>{
 const bend=X.map(x=>x<150?x*x/600:150*150/600+(x-150)*.5-(x-150)**2/300).map((y,i)=>[X[i],y,0,X[i]]);
 const box=[[10,0,0],[11,1,1]],hit={confirmed:true,bbox:box,distance_m:10,track_id:1},attn={confirmed:false,attention:true,bbox:box,distance_m:10};
 const frames=[frame(0),frame(1,{status:'candidate',candidates:[attn]}),frame(2,{obstacle:true,status:'obstacle',candidates:[hit],analysis_range_m:50}),
  frame(3,{status:'unknown'}),frame(4,{analysis_range_m:50,path:bend,path_info:{support_m:100}}),frame(5,{analysis_range_m:5})];
 const gt=(n,objects)=>({frame:n,time_s:n*.1,objects}),obj=(id,label,x)=>({id,name:id,label,parts:[[[x,0,0],[x+1,1,1]]]});
 const truth={frames:[gt(0,[]),gt(1,[obj('a','positive',10)]),gt(2,[obj('a','positive',10)]),gt(3,[obj('o','negative',10)]),gt(4,[obj('a','positive',30)]),gt(5,[obj('c','ambiguous',30)])]};
 const evaluation=evaluate(frames,truth,{pad:0}),lanes=T.buildLanes({frames,truth,evaluation});
 assert.deepEqual([...lanes.cells.decision],[1,3,4,2,1,1]);
 assert.deepEqual([...lanes.cells.truth],[0,3,3,1,3,2]);
 assert.deepEqual([...lanes.cells.attention],[0,1,0,0,0,0]);
 assert.deepEqual([...lanes.cells.axis],[0,0,0,0,1,0]);
 assert.deepEqual([...lanes.cells.fp],[0,0,0,0,0,0]);
 assert.deepEqual([...lanes.cells.fn],[0,1,0,0,2,0]);
 assert.deepEqual(lanes.indices.fn,[1,4]);
});
test('paint keeps the strongest code in every pixel and never hides a single frame',()=>{
 const cells=new Uint8Array(1000);cells[500]=1;cells[501]=2;cells[999]=1;
 const out=T.paint(cells,0,1000,100);
 assert.equal(out[50],2);assert.equal(out[99],1);assert.equal(out.reduce((a,v)=>a+(v?1:0),0),2);
 const zoomed=T.paint(cells,495,505,10);assert.deepEqual([...zoomed],[0,0,0,0,0,1,2,0,0,0]);
});
test('nearest and step find error runs from the current frame',()=>{
 const flagged=[3,4,5,20,21,50];
 assert.equal(T.nearest(flagged,19,2),20);assert.equal(T.nearest(flagged,12,2),null);assert.equal(T.nearest(flagged,40),50);
 assert.equal(T.step(flagged,0,1),3);assert.equal(T.step(flagged,4,1),20);assert.equal(T.step(flagged,50,1),null);
 assert.equal(T.step(flagged,50,-1),20);assert.equal(T.step(flagged,21,-1),20);assert.equal(T.step(flagged,3,-1),null);
});
test('zoom keeps the anchor in place, stays inside the run and can follow the playhead',()=>{
 let view={start:0,end:1000};
 view=T.zoomView(view,1000,4,.5);assert.deepEqual(view,{start:375,end:625});
 view=T.zoomView(view,1000,.25,.5);assert.deepEqual(view,{start:0,end:1000});
 view=T.zoomView({start:0,end:1000},1000,10,0);assert.deepEqual(view,{start:0,end:100});
 assert.deepEqual(T.panView({start:0,end:100},1000,-50),{start:0,end:100});
 assert.deepEqual(T.panView({start:900,end:1000},1000,50),{start:900,end:1000});
 assert.deepEqual(T.followView({start:0,end:100},1000,500),{start:450,end:550});
 assert.deepEqual(T.followView({start:0,end:100},1000,50),{start:0,end:100});
 assert.ok(T.zoomView({start:0,end:1000},1000,1e6).end-T.zoomView({start:0,end:1000},1000,1e6).start>=40);
});
test('lane near marks duplicates and shifted boxes; FP lane tells leftovers from live boxes',()=>{
 const b=x=>[[x,0,0],[x+1,1,1]],pred=(x,extra={})=>({confirmed:true,bbox:b(x),distance_m:x,track_id:x,...extra});
 const obj=(id,x)=>({id,name:id,label:'positive',parts:[b(x)]});
 const frames=[frame(0,{candidates:[pred(10),pred(10.3)],analysis_range_m:50}),frame(1,{candidates:[pred(11.3,{held:true,point_count:0})],analysis_range_m:50}),
  frame(2,{candidates:[pred(60,{held:true,point_count:0})],analysis_range_m:50}),frame(3,{candidates:[pred(70)],analysis_range_m:50})];
 const truth={frames:[0,1,2,3].map(n=>({frame:n,time_s:n*.1,objects:[obj('a',10)]}))};
 const evaluation=evaluate(frames,truth,{pad:0}),lanes=T.buildLanes({frames,truth,evaluation});
 assert.deepEqual([...lanes.cells.near],[1,2,0,0]);
 assert.deepEqual([...lanes.cells.fp],[0,0,2,1]);
 assert.deepEqual(T.LANES.find(l=>l.id==='near').errors,['DUP','SHIFT']);
});
