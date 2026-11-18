/* Free-flight camera: position is independent of the viewed cloud. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();
  else root.FreeCamera=factory();
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const clamp=(n,min,max)=>Math.max(min,Math.min(max,n));
  const length=v=>Math.hypot(...v);
  class FreeCamera {
    constructor(){
      this.fov=60;this.targetFov=60;this.speed=8;this.position=[0,0,0];this.yaw=0;this.pitch=0;this.mode='free';
      // Inertia: velocity eases toward the held direction, pending turn/zoom ease out.
      this.velocity=[0,0,0];this.pendingYaw=0;this.pendingPitch=0;
      this.accelTime=.14;this.brakeTime=.22;this.turnTime=.07;this.zoomTime=.08;
      this.teleport('perspective');
    }
    halt(){this.velocity=[0,0,0];this.pendingYaw=0;this.pendingPitch=0;this.targetFov=this.fov}
    basis(){
      const sy=Math.sin(this.yaw),cy=Math.cos(this.yaw),sp=Math.sin(this.pitch),cp=Math.cos(this.pitch);
      return {forward:[sy*cp,sp,-cy*cp],right:[cy,0,sy],up:[-sy*sp,cp,cy*sp]};
    }
    lookAt(position,target){
      this.halt();this.position=[...position];const d=target.map((n,i)=>n-position[i]),n=length(d);
      if(n>0){this.pitch=Math.asin(clamp(d[1]/n,-1,1));this.yaw=Math.abs(d[0])+Math.abs(d[2])<1e-8?0:Math.atan2(d[0],-d[2])}
    }
    teleport(preset){
      const poses={perspective:[[24,14,24],[0,-.6,-30]],cab:[[0,.1,1.5],[0,-.2,-45]],top:[[0,85,-47],[0,0,-47]]};
      if(!poses[preset])return;
      this.lookAt(...poses[preset]);this.mode=preset;
    }
    rotate(dx,dy){
      if(!dx&&!dy)return false;
      this.yaw=(this.yaw+dx*.003)%(Math.PI*2);
      this.pitch=clamp(this.pitch-dy*.003,-Math.PI/2+.001,Math.PI/2-.001);
      this.mode='free';return true;
    }
    changeFov(delta){this.fov=clamp(this.fov+delta*.04,20,110);this.targetFov=this.fov}
    /* Smoothed controls for the viewer loop; rotate/changeFov/move stay immediate. */
    turn(dx,dy){
      if(!dx&&!dy)return false;
      this.pendingYaw+=dx*.003;this.pendingPitch-=dy*.003;this.mode='free';return true;
    }
    zoom(delta){this.targetFov=clamp(this.targetFov+delta*.04,20,110)}
    fly(keys,dt){
      dt=Math.min(Math.max(dt||0,0),.1);if(dt<=0)return false;
      const f=Number(keys.has('KeyW'))-Number(keys.has('KeyS'));
      const r=Number(keys.has('KeyD'))-Number(keys.has('KeyA'));
      const vertical=Number(keys.has('KeyE'))-Number(keys.has('KeyQ'));
      const {forward,right}=this.basis();
      const direction=forward.map((n,i)=>n*f+right[i]*r+(i===1?vertical:0)),n=length(direction);
      const held=n>1e-8,boost=keys.has('ShiftLeft')||keys.has('ShiftRight')?4:1;
      const target=held?direction.map(x=>x/n*this.speed*boost):[0,0,0];
      const k=1-Math.exp(-dt/(held?this.accelTime:this.brakeTime));
      this.velocity=this.velocity.map((v,i)=>v+(target[i]-v)*k);
      if(!held&&length(this.velocity)<.03){this.velocity=[0,0,0];return false}
      this.position=this.position.map((p,i)=>p+this.velocity[i]*dt);
      this.mode='free';return true;
    }
    update(keys,dt){
      dt=Math.min(Math.max(dt||0,0),.1);
      let changed=this.fly(keys,dt);
      if(dt>0&&(Math.abs(this.pendingYaw)>1e-5||Math.abs(this.pendingPitch)>1e-5)){
        const k=1-Math.exp(-dt/this.turnTime),dy=this.pendingYaw*k,dp=this.pendingPitch*k;
        this.pendingYaw-=dy;this.pendingPitch-=dp;this.yaw=(this.yaw+dy)%(Math.PI*2);
        const pitch=clamp(this.pitch+dp,-Math.PI/2+.001,Math.PI/2-.001);
        if(pitch!==this.pitch+dp)this.pendingPitch=0;
        this.pitch=pitch;changed=true;
      }else{this.pendingYaw=0;this.pendingPitch=0}
      if(dt>0&&Math.abs(this.targetFov-this.fov)>.01){
        this.fov+=(this.targetFov-this.fov)*(1-Math.exp(-dt/this.zoomTime));changed=true;
      }else if(this.fov!==this.targetFov){this.fov=this.targetFov;changed=changed||dt>0}
      return changed;
    }
    move(keys,dt){
      const f=Number(keys.has('KeyW'))-Number(keys.has('KeyS'));
      const r=Number(keys.has('KeyD'))-Number(keys.has('KeyA'));
      const vertical=Number(keys.has('KeyE'))-Number(keys.has('KeyQ'));
      const {forward,right}=this.basis();
      const direction=forward.map((n,i)=>n*f+right[i]*r+(i===1?vertical:0));
      const n=length(direction);if(n<1e-8||dt<=0)return false;
      const boost=keys.has('ShiftLeft')||keys.has('ShiftRight')?4:1;
      const distance=this.speed*boost*Math.min(dt,.05);
      this.position=this.position.map((p,i)=>p+direction[i]/n*distance);
      this.mode='free';return true;
    }
    fitBounds(lo,hi,aspect){
      const center=lo.map((n,i)=>(n+hi[i])/2),half=length(hi.map((n,i)=>n-lo[i]))/2;
      const angle=Math.atan(Math.tan(this.fov*Math.PI/360)*Math.min(1,aspect));
      const distance=Math.max(5,half/Math.sin(angle)*1.08),{forward}=this.basis();
      this.halt();this.position=center.map((n,i)=>n-forward[i]*distance);this.mode='free';
    }
    viewMatrix(){
      const {forward,right:x,up:y}=this.basis(),z=forward.map(n=>-n),p=this.position;
      const dot=a=>a.reduce((s,n,i)=>s+n*p[i],0);
      return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x),-dot(y),-dot(z),1];
    }
  }
  return FreeCamera;
});
