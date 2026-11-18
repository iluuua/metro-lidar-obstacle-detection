const test=require('node:test'),assert=require('node:assert/strict');
const {frameFor,pathPoint}=require('../viewer/model-view.js');

test('organizer section follows the full 3D basis and preserves rail-head zero',()=>{
 const {envelopeOutline,vehicleCorners}=require('../viewer/model-view.js');
 const q=Math.sqrt(.5);
 const f={floor:[0,0,-9],section_frames:[[0,0,0,0,0,q,q,0,-q,q],[10,10,0,0,0,q,q,0,-q,q]],
  vehicle:{envelope_mode:'section',body_bottom:0,max_height:3},min_height_m:.02,
  envelope:[[0,-1.05,1.05,-1.05,1.05],[10,-1.05,1.05,-1.05,1.05]]};
 assert.deepEqual(pathPoint(f,5,0,0),[5,0,0]);
 const p=pathPoint(f,5,1,1);assert.ok(Math.abs(p[1])<1e-12);assert.ok(Math.abs(p[2]-2*q)<1e-12);
 assert.deepEqual(envelopeOutline(f,5),[[-1.05,0],[1.05,0],[1.05,3],[-1.05,3]]);
 assert.deepEqual(vehicleCorners(f,5),[]);
});

test('frame decisions replace the preview and unknown frames keep no old geometry',()=>{
 const base={frame:4,curve:[0,0,0],floor:[0,0,-1.4],obstacle:true,candidates:[{}],file:'preview.bin',full_file:'full.bin'};
 const result={frame:4,status:'unknown',obstacle:false,candidates:[]};
 const merged=frameFor(base,result);
 assert.equal(merged.floor,undefined);assert.equal(merged.curve,undefined);assert.equal(merged.obstacle,false);
 assert.equal(merged.file,base.file);assert.equal(merged.full_file,base.full_file);
 assert.equal(frameFor(base,null),base);
});
test('path is interpolated by arc distance and the envelope is normal to its tangent',()=>{
 const f={floor:[0,0,-1],path:[[0,0,-1,0],[3,4,-1,5],[6,8,-1,10]]};
 assert.deepEqual(pathPoint(f,2.5,0,1),[1.5,2,0]);
 const p=pathPoint(f,2.5,2,1),c=pathPoint(f,2.5,0,1);
 assert.ok(Math.abs(Math.hypot(p[0]-c[0],p[1]-c[1])-2)<1e-12);
 assert.ok(Math.abs((p[0]-c[0])*3+(p[1]-c[1])*4)<1e-12);
 assert.deepEqual(pathPoint(f,20),[6,8,-1]);
});
test('polynomial path and floor are still read',()=>{
 const f={floor:[.01,.02,-1.4],curve:[.1,.001,.2]};
 const p=pathPoint(f,10,1,.5);assert.ok(Math.abs(p[1]-2.3)<1e-12);
 assert.ok(Math.abs(p[2]-(.1+.046-1.4+.5))<1e-12);
});

test('geometry can continue beyond the prediction envelope',()=>{
 const {geometryLength}=require('../viewer/model-view.js');
 const frame={analysis_range_m:60,floor:[0,0,-1.4],path:[[0,0,-1.4,0],[120,10,-1.4,121]]};
 assert.equal(geometryLength(frame),121);assert.equal(frame.analysis_range_m,60);
 assert.ok(pathPoint(frame,100)[0]>60);
});

test('short section export does not freeze the far axis or flatten its height',()=>{
 const frame={floor:[0,0,-1],section_frames:[[0,0,0,-1,0,1,0,0,0,1],[10,10,0,-.9,0,1,0,0,0,1]],
  path:[[0,0,-1,0],[10,0,-.9,10],[20,1,-.8,20.05]]};
 const a=pathPoint(frame,10),b=pathPoint(frame,10.001),far=pathPoint(frame,20);
 assert.ok(Math.hypot(...a.map((x,i)=>x-b[i]))<.002);
 assert.ok(far[0]>19&&far[2]>-.81);
});


test('envelope outline uses asymmetric model bounds and vehicle height',()=>{
 const {envelopeOutline}=require('../viewer/model-view.js');
 const f={envelope:[[0,-1.5,1.4,-1.1,1],[10,-1.7,1.6,-1.3,1.2]],vehicle:{body_bottom:1.172,max_height:3.87}};
 const outline=envelopeOutline(f,5);
 assert.deepEqual(outline[4],[1.5,3.87]);assert.deepEqual(outline[5],[-1.6,3.87]);
 assert.equal(outline[3][1],1.172);
});

test('wagon is a rigid chord with the specified length, width and pivot base',()=>{
 const {vehicleCorners}=require('../viewer/model-view.js'),R=300;
 const f={floor:[0,0,0],vehicle:{car_length:20.12,car_width:2.684,bogie_base:12.6,body_bottom:1.172,max_height:3.87},path:[]};
 for(let s=0;s<=120;s+=.05)f.path.push([R*Math.sin(s/R),R*(1-Math.cos(s/R)),0,s]);
 const c=vehicleCorners(f,30),distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
 assert.ok(Math.abs(distance(c[0],c[1])-20.12)<1e-8);
 assert.ok(Math.abs(distance(c[0],c[2])-2.684)<1e-8);
 const center=[(c[0][0]+c[3][0])/2,(c[0][1]+c[3][1])/2];
 const inward=R-Math.hypot(center[0],center[1]-R);
 assert.ok(Math.abs(inward-(R-Math.sqrt(R*R-6.3*6.3)))<1e-5);
});
