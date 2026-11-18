"""Shadow evaluation of continuous, signed, fading track evidence.

Does not change native results. Legacy saved candidates exclude exterior
returns. The opt-in native observation band also exports signed point depths.
The normal CDF is a smooth weight, not a calibrated collision probability.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Accumulator:
    tau: float = 2.
    reference: float = .3
    threshold: float = .2
    value: float = 0.
    last_time: float | None = None
    hits: int = 0

    def observe(self, time, z, quality):
        dt = .1 if self.last_time is None else time-self.last_time
        if self.last_time is not None and (dt <= 0 or dt > .5):
            self.value = 0.
            self.hits = 0
            dt = .1
        self.last_time = time
        weight = .5*(1+math.erf(z/math.sqrt(2)))
        decay = math.exp(-dt/self.tau)
        self.value = max(0., self.value*decay+self.tau*(1-decay)*quality*(weight-self.reference))
        self.hits += quality > 0
        return self.hits >= 3 and self.value >= self.threshold


def observation(c):
    """Use native signed point depth when available, legacy cluster surrogate otherwise."""
    unique = c.get('unique_point_count',0)
    if unique < 2 or c.get('held') or c.get('wall_like') or c.get('unverified'):
        return None
    # These bounds are published in the candidate's path coordinates.
    lo,hi = c.get('bbox',([0,0,0],[0,0,0]))
    if hi[0]-lo[0] >= 3 or hi[1]-lo[1] >= 1.5 or hi[2]-lo[2] >= 2.5:
        return None
    sigma = max(c.get('sigma_u',0),c.get('sigma_h',0),.02)
    z = c.get('soft_z',c.get('margin_m',0)/sigma)
    quality = min(unique/4,1.)*min(1.,.15/sigma)
    return z,quality


def evaluate(model, parameters):
    states = [{} for _ in parameters]
    extra = [[] for _ in parameters]
    results = [[] for _ in parameters]
    frames=model['frames']
    for frame in frames:
        # One vote per track per frame, even when a track has several fragments.
        observations={}
        for c in frame.get('candidates',[]):
            obs=observation(c)
            tid=c.get('track_id',-1)
            if obs is None or tid < 0:
                continue
            if tid not in observations or obs[0]*obs[1] > observations[tid][1][0]*observations[tid][1][1]:
                observations[tid]=(c,obs)
        for variant,(params,tracks) in enumerate(zip(parameters,states)):
            newly=[]
            for tid,(c,(z,q)) in observations.items():
                acc=tracks.setdefault(tid,Accumulator(**params))
                if acc.observe(frame['time_s'],z,q):
                    newly.append(dict(c,confirmed=True,soft_shadow=True))
            if newly and not frame.get('obstacle'):
                extra[variant].append(frame['frame'])
            results[variant].append(dict(frame=frame['frame'],status='obstacle' if newly else 'clear',candidates=newly))
            if frame['frame']%100 == 0:
                for tid in list(tracks):
                    if frame['time_s']-tracks[tid].last_time > .5:
                        del tracks[tid]
    return extra,results


def main():
    import argparse
    import sys
    sys.path.insert(0,str(ROOT/'eval'))
    from score_synthetic import score
    parameters=[dict(tau=tau,reference=ref,threshold=threshold)
                for tau in [1.,3.] for ref in [.3,.5] for threshold in [.1,.3,.6]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth', type=Path, help='optional user-supplied truth JSON')
    parser.add_argument('--truth-run', help='viewer run to which --truth applies')
    args = parser.parse_args()
    if bool(args.truth) != bool(args.truth_run):
        parser.error('--truth and --truth-run must be provided together')
    reports=[]
    gt=json.loads(args.truth.read_text(encoding='utf-8')) if args.truth else None
    for path in sorted((ROOT/'viewer/data').glob('*/model.tunnel.json')):
        model=json.loads(path.read_text(encoding='utf-8'))
        extras,results=evaluate(model,parameters)
        run=dict(id=path.parent.name,frames=len(model['frames']),variants=[])
        for params,added,result in zip(parameters,extras,results):
            entry=dict(parameters=params,added_alarm_frames=len(added),added_frame_ids=added)
            if gt is not None and path.parent.name==args.truth_run:
                # Score only the proposed channel, avoiding masking by baseline alarms.
                entry['new_channel_score']=score({r['frame']:r for r in result},gt,verbose=False)
            run['variants'].append(entry)
        reports.append(run)
        print(run['id'],len(model['frames']),'added', [len(a) for a in extras],flush=True)
    dest=ROOT/'output/soft_threat'
    dest.mkdir(exist_ok=True)
    (dest/'shadow.json').write_text(json.dumps(dict(note=__doc__,parameters=parameters,runs=reports),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
