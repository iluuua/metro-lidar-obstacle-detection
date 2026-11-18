const test=require('node:test'),assert=require('node:assert/strict');
const {evaluate}=require('../viewer/evaluation.js');
const b=(x,y=0)=>[[x,y,0],[x+1,y+1,1]];
const obj=(id,x,label='positive')=>({id,name:id,label,parts:[b(x)]});
const frame=(candidates,frame=0)=>({frame,time_s:frame*.1,status:'obstacle',candidates});
const pred=(x,track_id=1)=>({confirmed:true,bbox:b(x),distance_m:x,track_id});
const gt=(objects,frame=0)=>({frame,time_s:frame*.1,objects});
test('one prediction cannot detect two overlapping objects',()=>{
 const r=evaluate([frame([pred(10)])],{frames:[gt([obj('a',10),obj('b',10)])]},{pad:0});
 assert.equal(r.counts.tp,1);assert.equal(r.counts.fn,1);assert.equal(r.counts.fp,0);
});
test('matching reassigns a broad detection to preserve both matches',()=>{
 const broad={...pred(10),bbox:[[10,0,0],[13,1,1]]};
 const r=evaluate([frame([broad,pred(10,2)])],{frames:[gt([obj('a',10),obj('b',12)])]},{pad:0});
 assert.equal(r.counts.tp,2);assert.equal(r.counts.fn,0);assert.equal(r.counts.fp,0);
});
test('unmatched and exterior predictions are FP, attention is not TP',()=>{
 const r=evaluate([frame([pred(10),pred(20,2),{...pred(30),confirmed:false}])],{frames:[gt([obj('outside',10,'negative'),obj('missed',30)])]},{pad:0});
 assert.equal(r.counts.fp,2);assert.equal(r.counts.fn,1);assert.equal(r.counts.tp,0);
});
test('ambiguous policy has three explicit outcomes',()=>{
 const frames=[frame([pred(10)])],truth={frames:[gt([obj('edge',10,'ambiguous')])]};
 assert.equal(evaluate(frames,truth,{ambiguous:'ignore'}).counts.ignored,1);
 assert.equal(evaluate(frames,truth,{ambiguous:'positive'}).counts.tp,1);
 assert.equal(evaluate(frames,truth,{ambiguous:'negative'}).counts.fp,1);
});

test('boundary caution has separate coverage and navigable misses, without collision TP',()=>{
 const frames=[frame([{...pred(10),confirmed:false,attention:true}],0),frame([],1)];
 const truth={frames:[gt([obj('edge',10,'ambiguous')],0),gt([obj('edge',10,'ambiguous')],1)]};
 const r=evaluate(frames,truth);
 assert.equal(r.counts.tp,0);assert.equal(r.counts.fn,0);assert.equal(r.counts.fp,0);
 assert.equal(r.boundary.counts.covered,1);assert.equal(r.boundary.counts.attention,1);assert.equal(r.boundary.counts.missed,1);
 assert.equal(r.errors[0].type,'WFN');assert.equal(r.errors[0].frame,1);
});
test('one caution cannot cover two boundary cubes and false cautions are separate',()=>{
 const r=evaluate([frame([{...pred(10),confirmed:false,attention:true},{...pred(30,2),confirmed:false,attention:true}])],{frames:[gt([obj('a',10,'ambiguous'),obj('b',10,'ambiguous')])]});
 assert.equal(r.boundary.counts.covered,1);assert.equal(r.boundary.counts.missed,1);assert.equal(r.boundary.counts.falseAttention,1);
 assert.equal(r.counts.fp,0);assert.ok(r.errors.some(e=>e.type==='WFP'));
});
test('parts of a person count as a single object and nearby FN is navigable',()=>{
 const person={...obj('person',10),parts:[b(10),b(12)]};
 const r=evaluate([frame([pred(12)],0),frame([],1),frame([],2)],{frames:[gt([person],0),gt([person],1),gt([person],2)]});
 assert.equal(r.counts.tp,1);assert.equal(r.counts.fn,2);assert.equal(r.total,1);assert.equal(r.detected,1);
 assert.equal(r.episodes.length,1);assert.equal(r.episodes[0].startFrame,1);assert.equal(r.episodes[0].endFrame,2);
});
test('range and padding apply equally to all versions, missing results count as missed objects',()=>{
 const r=evaluate([frame([pred(50)])],{frames:[gt([obj('far',50)]),gt([obj('absent-frame',10)],1)]},{maxRange:20});
 assert.equal(r.counts.fp,0);assert.equal(r.counts.fn,1);assert.equal(r.counts.evaluatedFrames,2);assert.equal(r.counts.missingFrames,1);
 assert.equal(evaluate([frame([pred(11.1)])],{frames:[gt([obj('a',10)])]},{pad:.2}).counts.tp,1);
});
test('a miss beyond the detector\'s own verified range is counted separately from a miss inside it',()=>{
 const f=(range)=>({...frame([]),analysis_range_m:range});
 const near=evaluate([f(40)],{frames:[gt([obj('a',30)])]}),far=evaluate([f(40)],{frames:[gt([obj('a',90)])]});
 assert.equal(near.counts.fn,1);assert.equal(near.counts.fnBeyond,0);assert.equal(near.errors[0].beyondVerified,false);assert.equal(near.perFrame.get(0).fnBeyond,0);
 assert.equal(far.counts.fn,1);assert.equal(far.counts.fnBeyond,1);assert.equal(far.errors[0].beyondVerified,true);assert.equal(far.perFrame.get(0).fnBeyond,1);
});
test('false positives on known exterior objects are told apart from unexplained ones',()=>{
 const r=evaluate([frame([pred(10),pred(20,2)])],{frames:[gt([obj('outside',10,'negative')])]},{pad:0});
 assert.equal(r.counts.fp,2);assert.equal(r.counts.fpExterior,1);
 assert.deepEqual(r.errors.map(e=>e.exterior),[true,false]);
});
test('a second box on an already detected target is a duplicate, not a false alarm',()=>{
 const r=evaluate([frame([pred(10),pred(10.3,2)])],{frames:[gt([obj('a',10)])]},{pad:0});
 assert.equal(r.counts.tp,1);assert.equal(r.counts.dup,1);assert.equal(r.counts.fp,0);assert.equal(r.counts.fn,0);
 assert.equal(r.errors[0].type,'DUP');assert.equal(r.errors[0].name,'a');assert.equal(r.perFrame.get(0).dup,1);
});
test('a box shifted off its target within the tolerance is counted on its own, beyond it is FP plus FN',()=>{
 const shifted=evaluate([frame([{...pred(11.3),held:true,point_count:0}])],{frames:[gt([obj('a',10)])]},{pad:0});
 assert.equal(shifted.counts.shifted,1);assert.equal(shifted.counts.tp,0);assert.equal(shifted.counts.fp,0);assert.equal(shifted.counts.fn,0);
 assert.equal(shifted.errors[0].type,'SHIFT');assert.equal(shifted.errors[0].held,true);assert.equal(shifted.detected,1);
 const far=evaluate([frame([pred(10+1+DetectorEvaluation_NEAR()+.5)])],{frames:[gt([obj('a',10)])]},{pad:0});
 assert.equal(far.counts.shifted,0);assert.equal(far.counts.fp,1);assert.equal(far.counts.fn,1);
});
test('a held box without points is marked as a leftover; a fresh one is not',()=>{
 const r=evaluate([frame([{...pred(40),held:true,point_count:0},{...pred(60,2),held:true,point_count:30},pred(80,3)])],{frames:[gt([obj('a',10)])]},{pad:0});
 assert.equal(r.counts.fp,3);assert.equal(r.counts.fpStale,1);
 assert.deepEqual(r.errors.filter(e=>e.type==='FP').map(e=>e.stale),[true,false,false]);
});
test('a box near a disputed cube is ignored with it when disputed cubes are set aside',()=>{
 const r=evaluate([frame([pred(11.2)])],{frames:[gt([obj('edge',10,'ambiguous')])]},{pad:0,ambiguous:'ignore'});
 assert.equal(r.counts.ignored,1);assert.equal(r.counts.fp,0);
});
function DetectorEvaluation_NEAR(){return require('../viewer/evaluation.js').NEAR}
