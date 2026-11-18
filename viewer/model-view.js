/* Shared model-frame and path geometry helpers; no WebGL or DOM dependency. */
(function(root){
 'use strict';
 function frameFor(base,result){
  if(!result)return base;
  const out={...result};
  for(const key of ['file','full_file','full_bounds','display_points','full_points','raw_points'])if(key in base)out[key]=base[key];
  // Early-exit results (status unknown) may omit the input point count; the
  // decoded cloud is the same one the recording stored for this frame.
  if(out.points==null&&base.points!=null)out.points=base.points;
  if(!Array.isArray(out.candidates))out.candidates=[];
  return out;
 }
 function pathPoint(frame,distance,lateral=0,height=0){
  const fl=frame.floor;
  // the far axis beyond the last exported section continues along the full path (Igor, 19.10)
  if(frame.section_frames?.length>=2&&(!frame.path?.length||distance<=frame.section_frames[frame.section_frames.length-1][0])){
   const frames=frame.section_frames;let lo=0,hi=frames.length-1;
   distance=Math.max(frames[0][0],Math.min(frames[hi][0],distance));
   while(hi-lo>1){const mid=(lo+hi)>>1;if(frames[mid][0]<distance)lo=mid;else hi=mid}
   const a=frames[lo],b=frames[hi],t=(distance-a[0])/(b[0]-a[0]||1);
   return [1,2,3].map(k=>a[k]+t*(b[k]-a[k])+lateral*(a[k+3]+t*(b[k+3]-a[k+3]))+height*(a[k+6]+t*(b[k+6]-a[k+6])));
  }
  if(frame.path?.length>=2){
   const path=frame.path;let lo=0,hi=path.length-1;
   distance=Math.max(path[0][3],Math.min(path[hi][3],distance));
   while(hi-lo>1){const mid=(lo+hi)>>1;if(path[mid][3]<distance)lo=mid;else hi=mid}
   const a=path[lo],b=path[hi],t=(distance-a[3])/(b[3]-a[3]||1);
   const dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy)||1;
   const phi=frame.path_info?.phi??0,ln=Math.hypot(1,phi);
   const x=a[0]+(b[0]-a[0])*t-lateral*dy/n/ln+height*dy/n*phi/ln;
   const y=a[1]+(b[1]-a[1])*t+lateral*dx/n/ln-height*dx/n*phi/ln;
   const z=Number.isFinite(a[2])&&Number.isFinite(b[2])?a[2]+(b[2]-a[2])*t:fl[0]*x+fl[1]*y+fl[2];
   return [x,y,z+lateral*phi/ln+height/ln];
  }
  const cv=frame.curve||[0,0,0],x=distance,y=cv[0]*x+cv[1]*x*x+(cv[2]||0)+lateral;
  return [x,y,fl[0]*x+fl[1]*y+fl[2]+height];
 }
 function geometryLength(frame){return frame?.path?.length?frame.path[frame.path.length-1][3]:(frame?.analysis_range_m||60)}
 function envelopeOutline(frame,s){
  const samples=frame.envelope;
  let lower=-(frame.half_width_m||1.45),upper=-lower,lowLower=-1,lowUpper=1;
  if(samples?.length){let i=0;while(i+1<samples.length&&samples[i+1][0]<s)i++;const a=samples[i],b=samples[Math.min(i+1,samples.length-1)],t=Math.max(0,Math.min(1,(s-a[0])/(b[0]-a[0]||1)));[lower,upper,lowLower,lowUpper]=[1,2,3,4].map(k=>a[k]+(b[k]-a[k])*t)}
  const bottom=frame.vehicle?.body_bottom??.85,top=frame.vehicle?.max_height??3.5,min=frame.min_height_m??.35;
  if(frame.vehicle?.envelope_mode==='section')return [[lower,0],[upper,0],[upper,top],[lower,top]];
  return [[lowLower,min],[lowUpper,min],[lowUpper,bottom],[upper,bottom],[upper,top],[lower,top],[lower,bottom],[lowLower,bottom]];
 }
 function vehicleCorners(frame,s){
  if(!frame.vehicle||!frame.path||frame.vehicle.envelope_mode==='section')return [];
  const v=frame.vehicle;
  // Bogie base is the straight pivot-to-pivot distance, not arc length.
  let lo=v.bogie_base/2,hi=Math.min(s,geometryLength(frame)-s);
  if(hi<lo)return [];
  const span=h=>{const a=pathPoint(frame,s-h),b=pathPoint(frame,s+h);return Math.hypot(b[0]-a[0],b[1]-a[1])};
  if(span(hi)<v.bogie_base)return [];
  for(let i=0;i<24;i++){const mid=(lo+hi)/2;if(span(mid)<v.bogie_base)lo=mid;else hi=mid}
  const a=pathPoint(frame,s-hi),b=pathPoint(frame,s+hi);
  const n=Math.hypot(b[0]-a[0],b[1]-a[1]),dx=(b[0]-a[0])/n,dy=(b[1]-a[1])/n;
  const center=[(a[0]+b[0])/2,(a[1]+b[1])/2],corners=[];
  for(let i=0;i<8;i++){const along=(i&1?1:-1)*v.car_length/2,across=(i&2?1:-1)*v.car_width/2;
   const x=center[0]+along*dx-across*dy,y=center[1]+along*dy+across*dx,fl=frame.floor;
   corners.push([x,y,fl[0]*x+fl[1]*y+fl[2]+(i&4?v.max_height:v.body_bottom)]);
  }return corners;
 }
 const api={frameFor,pathPoint,geometryLength,envelopeOutline,vehicleCorners};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ModelView=api;
})(typeof window!=='undefined'?window:this);
