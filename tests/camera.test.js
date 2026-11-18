const test=require('node:test');
const assert=require('node:assert/strict');
const FreeCamera=require('../viewer/camera.js');
const near=(a,b,eps=1e-8)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
test('looking around and changing FOV preserve position',()=>{
 const c=new FreeCamera(),p=[...c.position];c.rotate(120,-50);assert.deepEqual(c.position,p);
 c.changeFov(500);assert.deepEqual(c.position,p);near(c.fov,80);
 c.changeFov(1e6);near(c.fov,110);c.changeFov(-1e6);near(c.fov,20);
});
test('WASD follows camera direction and diagonal speed is normalized',()=>{
 const a=new FreeCamera(),b=new FreeCamera();a.teleport('cab');b.teleport('cab');const p=[...a.position];
 a.move(new Set(['KeyW']),.04);b.move(new Set(['KeyW','KeyD']),.04);
 assert.ok(a.position[2]<p[2]);near(a.position[0],p[0]);near(distance(a.position,p),distance(b.position,p));
});
test('opposite movement cancels and Shift multiplies travel by four',()=>{
 const a=new FreeCamera(),b=new FreeCamera(),p=[...a.position];
 assert.equal(a.move(new Set(['KeyW','KeyS','KeyA','KeyD']),.04),false);assert.deepEqual(a.position,p);
 a.move(new Set(['KeyE']),.04);b.move(new Set(['KeyE','ShiftLeft']),.04);near(distance(b.position,p),4*distance(a.position,p));
});
test('top and cab are teleports that allow immediate free flight',()=>{
 for(const preset of ['top','cab']){
  const c=new FreeCamera();c.teleport(preset);assert.ok(c.viewMatrix().every(Number.isFinite));
  const p=[...c.position];c.move(new Set(['KeyD']),.04);assert.notDeepEqual(c.position,p);assert.equal(c.mode,'free');
 }
});
test('turning changes the next movement direction rather than orbiting',()=>{
 const c=new FreeCamera();c.teleport('cab');c.pitch=0;c.rotate(Math.PI/2/.003,0);const p=[...c.position];
 c.move(new Set(['KeyW']),.04);assert.ok(c.position[0]>p[0]);near(c.position[2],p[2]);
});
test('fit is one-shot and no held input leaves the position unchanged',()=>{
 const c=new FreeCamera();c.fitBounds([-10,-5,-200],[10,5,0],1.5);const p=[...c.position];
 for(let i=0;i<60;i++)c.move(new Set(),1/60);assert.deepEqual(c.position,p);
 c.move(new Set(['KeyW']),.04);assert.notDeepEqual(c.position,p);
});
test('a long frame does not teleport the camera',()=>{
 const c=new FreeCamera(),p=[...c.position];c.move(new Set(['KeyW']),50);near(distance(c.position,p),c.speed*.05);
});
