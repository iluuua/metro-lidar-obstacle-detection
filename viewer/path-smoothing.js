/* Presentation geometry only: detector results and cloud coordinates stay intact. */
(function(root){
 'use strict';
 const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 function sample(path,x,column){
  let lo=0,hi=path.length-1;
  x=clamp(x,path[0][0],path[hi][0]);
  while(hi-lo>1){const mid=(lo+hi)>>1;if(path[mid][0]<x)lo=mid;else hi=mid}
  const a=path[lo],b=path[hi],t=(x-a[0])/(b[0]-a[0]||1);
  return a[column]+t*(b[column]-a[column]);
 }
 function slope(path,x,column){
  const n=path.length;
  if(n<3)return (path[n-1][column]-path[0][column])/(path[n-1][0]-path[0][0]||1);
  const derivative=i=>i===0?(-3*path[0][column]+4*path[1][column]-path[2][column])/(path[2][0]-path[0][0]):
   i===n-1?(3*path[n-1][column]-4*path[n-2][column]+path[n-3][column])/(path[n-1][0]-path[n-3][0]):
   (path[i+1][column]-path[i-1][column])/(path[i+1][0]-path[i-1][0]);
  let lo=0,hi=n-1;
  x=clamp(x,path[0][0],path[hi][0]);
  while(hi-lo>1){const mid=(lo+hi)>>1;if(path[mid][0]<x)lo=mid;else hi=mid}
  const t=(x-path[lo][0])/(path[hi][0]-path[lo][0]);
  return derivative(lo)*(1-t)+derivative(hi)*t;
 }
 function transported(path,x,ds,column){
  // Match the core's transport: preserve mounting offset/heading, advance shape.
  return sample(path,x+ds,column)-(sample(path,ds,column)-path[0][column])
   -(slope(path,ds,column)-slope(path,0,column))*x;
 }
 function build(frames,{tau=.35,release=.10,snap=.50,maxGap=.5}={}){
  if(!(tau>0&&release>=0&&snap>release&&maxGap>0))throw Error('Invalid path smoothing settings');
  const out=new Map();let previous=null;
  for(const frame of frames){
   const path=frame.path,dt=previous?frame.time_s-previous.time_s:0;
   if(frame.status==='unknown'||!Array.isArray(path)||path.length<2){previous=null;out.set(frame.frame,frame);continue}
   const ds=frame.path_info?.odometry_ds;
   if(!previous||!(dt>0&&dt<=maxGap)||!Number.isFinite(ds)){previous=frame;out.set(frame.frame,frame);continue}
   const base=1-Math.exp(-dt/tau);
   const gainFor=residual=>{const t=clamp((residual-release)/(snap-release),0,1);return base+(1-base)*t*t*(3-2*t)};
   const limit=Math.min(frame.envelope_range_m||0,(previous.envelope_range_m||0)-ds);
   const smoothed=path.map(point=>{
    const [x,y,z]=point;
    if(x+ds<previous.path[0][0]||x+ds>previous.path[previous.path.length-1][0]||point[3]>limit)return point.slice();
    const py=transported(previous.path,x,ds,1),pz=transported(previous.path,x,ds,2);
    // Fade the optional display correction out before the supported range
    // ends. A hard cutoff here made a lateral step within a single curve.
    const taper=clamp((limit-point[3])/20,0,1),weight=taper*taper*(3-2*taper);
    const a=1-(1-gainFor(Math.hypot(y-py,z-pz)))*weight;
    return [x,py+a*(y-py),pz+a*(z-pz),point[3]];
   });
   // Banking noise moves the upper corners even when the centre line is steady.
   const rawPhi=frame.path_info?.phi||0,previousPhi=previous.display_phi??previous.path_info?.phi??0;
   const phi=previousPhi+gainFor(3*Math.abs(rawPhi-previousPhi))*(rawPhi-previousPhi);
   const sections=frame.section_frames?.map(section=>{
    const x=section[1],dy=slope(smoothed,x,1),norm=Math.hypot(1,dy),ln=Math.hypot(1,phi);
    const tx=1/norm,ty=dy/norm,l=[-ty/ln,tx/ln,phi/ln];
    return [section[0],x,sample(smoothed,x,1),sample(smoothed,x,2),...l,ty*l[2],-tx*l[2],tx*l[1]-ty*l[0]];
   });
   const geometry={...frame,path:smoothed,display_phi:phi,...(sections?{section_frames:sections}:{})};
   out.set(frame.frame,geometry);previous=geometry;
  }
  return out;
 }
 const api={build};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.PathSmoothing=api;
})(typeof window!=='undefined'?window:this);
