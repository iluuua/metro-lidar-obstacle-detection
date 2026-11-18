/* Apply an optional, separately validated experiment without mutating baseline. */
(function(root){
 'use strict';
 function apply(frame,indices){
  if(!indices?.length||frame.status==='unknown')return frame;
  const selected=new Set(indices),candidates=frame.candidates.map((c,i)=>
   selected.has(i)&&!c.confirmed?{...c,confirmed:true,soft_confirmation:true}:c);
  if(!candidates.some(c=>c.soft_confirmation))return frame;
  const distances=candidates.filter(c=>c.confirmed&&Number.isFinite(c.distance_m)).map(c=>c.distance_m);
  if(frame.obstacle&&Number.isFinite(frame.distance_m))distances.push(frame.distance_m);
  return {...frame,candidates,obstacle:true,status:'obstacle',soft_confirmation:true,
   distance_m:distances.length?Math.min(...distances):frame.distance_m};
 }
 const api={apply};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.SoftConfirmation=api;
})(typeof window!=='undefined'?window:this);
