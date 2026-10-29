import json,math,statistics as st
from pathlib import Path
P=Path(__file__).resolve().parent;run=Path((P/'latest_run.txt').read_text().strip());rows=json.loads((run/'result/trace.json').read_text());out=[]
def quantile(a,p):
 a=sorted(a);return a[round((len(a)-1)*p)] if a else None
def contacts(forces,threshold):
 state=None;history=[]
 for i,f in enumerate(forces):
  if i and (f>threshold)==(forces[i-1]>threshold):state=f>threshold
  history.append(state)
 return history
for seg in range(2):
 allrows=rows[250*seg:250*(seg+1)];samples=allrows[50:]
 for foot,name in enumerate(['LH','RH','LF','RF']):
  force=[r['contact_force_z'][foot] for r in allrows];contact=contacts(force,5);runs=[];start=0
  for i in range(1,len(contact)+1):
   if i==len(contact) or contact[i]!=contact[start]:
    runs.append(dict(contact=contact[start],start=start,end=i,duration=(i-start)*.02));start=i
  swings=[]
  for k,x in enumerate(runs):
   if x['contact'] is False and x['start']>=50 and k>0 and k+1<len(runs) and runs[k-1]['contact'] is True and runs[k+1]['contact'] is True:
    rr=allrows[x['start']:x['end']];swings.append(dict(start=x['start']*.02,end=x['end']*.02,duration=x['duration'],peak_gap_mm=max(r['gap_mm'][foot] for r in rr)))
  landings=[x['start']*.02 for k,x in enumerate(runs) if x['contact'] is True and x['start']>=50 and k and runs[k-1]['contact'] is False]
  periods=[b-a for a,b in zip(landings,landings[1:])]
  gaps=[r['gap_mm'][foot] for r in samples]
  result=dict(direction=['left','right'][seg],foot=name,complete_swings=swings,landing_periods_s=periods,period_mean=st.mean(periods) if periods else None,period_cv=st.pstdev(periods)/st.mean(periods) if periods else None,swings_at_least_10mm=sum(x['peak_gap_mm']>=10 for x in swings),longest_air_s=max([max(0,x['end']-max(50,x['start']))*.02 for x in runs if x['contact'] is False]or[0]),max_gap_mm=max(gaps),mean_gap_mm=st.mean(gaps),sensitivity={})
  for threshold in [1,5,10]:
   c=contacts(force,threshold);ids=[i for i in range(50,250) if c[i] is True and force[i]>threshold]
   speeds=[math.hypot(*allrows[i]['foot_bottom_velocity_world'][foot][:2]) for i in ids];center=[math.hypot(*allrows[i]['foot_center_velocity_world'][foot][:2]) for i in ids]
   result['sensitivity'][threshold]=dict(contact_fraction=len(ids)/200,slip_mean=st.mean(speeds) if speeds else None,slip_p95=quantile(speeds,.95),slip_over_05_fraction=st.mean(x>.05 for x in speeds) if speeds else None,slip_distance_m=sum(speeds)*.02,center_speed_mean=st.mean(center) if center else None)
  out.append(result)
(run/'result/gait_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
