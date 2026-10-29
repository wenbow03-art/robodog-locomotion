"""Summarize paired policy tracking, validate URDF FK, and plot nominal traces."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def axis_rotation(axis, angles):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    a=np.asarray(angles)
    return np.eye(3)+np.sin(a)[...,None,None]*K+(1-np.cos(a))[...,None,None]*(K@K)

def origin(node):
    o=node.find('origin')
    xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
    rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
    R=axis_rotation([0,0,1],rpy[2])@axis_rotation([0,1,0],rpy[1])@axis_rotation([1,0,0],rpy[0])
    return xyz,R

def forward_kinematics(xml,names,feet,q):
    # Outputs the collision sphere centers in base_link coordinates.
    shape=q.shape[:-1];flat=q.reshape(-1,12);count=len(flat)
    parent={j.find('child').get('link'):j for j in xml.findall('joint')}
    outputs=[]
    for foot in feet:
        chain=[];link=foot
        while link!='base_link':
            joint=parent[link];chain.append(joint);link=joint.find('parent').get('link')
        R=np.broadcast_to(np.eye(3),(count,3,3)).copy();p=np.zeros((count,3))
        for joint in reversed(chain):
            xyz,rot=origin(joint);p+=np.einsum('nij,j->ni',R,xyz);R=R@rot
            if joint.get('type')!='fixed':
                assert joint.get('type')=='revolute'
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                R=R@axis_rotation(axis,flat[:,names.index(joint.get('name'))])
        xyz,_=origin(xml.find("link[@name='%s']/collision"%foot))
        outputs.append(p+np.einsum('nij,j->ni',R,xyz))
    return np.stack(outputs,axis=1).reshape(*shape,4,3)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);opts=p.parse_args();run=opts.run
    m=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'completion.json').read_text())
    assert complete['completed'] and complete['checkpoint_unchanged']
    for f,h in m['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    d=np.load(run/'traces.npz');s=np.load(run/'substeps.npz')['data']
    xml=ET.parse(ROOT/'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf').getroot()
    actual_fk=forward_kinematics(xml,m['dof_names'],m['feet'],d['q'])
    target_fk=forward_kinematics(xml,m['dof_names'],m['feet'],d['target'])
    valid=d['valid'];fk_error=np.linalg.norm(actual_fk-d['foot_base'],axis=-1)[valid]
    fk_max_mm=float(fk_error.max()*1000)
    assert fk_max_mm<1.,fk_max_mm
    np.savez_compressed(run/'kinematics.npz',actual_fk=actual_fk,target_fk=target_fk)
    hard=np.array([[float(xml.find("joint[@name='%s']/limit"%n).get(k)) for k in ['lower','upper']] for n in m['dof_names']])
    limits=np.array(m['torque_limits_Nm']);names=['Left hind','Right hind','Left front','Right front']
    groups=[('normal_nominal',[0]),('normal_all',list(range(9))),('mirror_nominal',[9]),('mirror_all',list(range(9,18)))]
    reports=[]
    for mode,ids in groups:
        mask=d['valid'][150:,ids];physmask=np.repeat(valid[:,ids],20,axis=0)[3000:]
        q=s[3000:,ids,0][physmask];dq=s[3000:,ids,1][physmask]
        target=s[3000:,ids,2][physmask];raw=s[3000:,ids,3][physmask]
        f=d['fz'][150:,ids][mask];g=d['gap'][150:,ids][mask]
        actual=actual_fk[150:,ids][mask];desired=target_fk[150:,ids][mask]
        bodyvel=d['foot_bottom_velocity'][150:,ids][mask]
        q20=d['q'][150:,ids][mask];t20=d['target'][150:,ids][mask]
        legs=[]
        for i,name in enumerate(names):
            js=slice(i*3,(i+1)*3);air=f[:,i]<=1.
            force=f[:,i];slip=np.linalg.norm(bodyvel[:,i,:2],axis=-1)
            row=dict(leg=name,contact_pct=float(100*(force>1).mean()),mean_force_N=float(force.mean()),
                gap_p95_mm=float(np.percentile(g[:,i],95)*1000),gap_max_mm=float(g[:,i].max()*1000),
                target_joint_p95_p05_deg=((np.percentile(target[:,js],95,axis=0)-np.percentile(target[:,js],5,axis=0))*180/np.pi).tolist(),
                actual_joint_p95_p05_deg=((np.percentile(q[:,js],95,axis=0)-np.percentile(q[:,js],5,axis=0))*180/np.pi).tolist(),
                joint_tracking_mae_deg=(np.abs(target[:,js]-q[:,js]).mean(0)*180/np.pi).tolist(),
                joint_tracking_p95_error_deg=(np.percentile(np.abs(target[:,js]-q[:,js]),95,axis=0)*180/np.pi).tolist(),
                torque_clipping_pct=(100*(np.abs(raw[:,js])>limits[js]+1e-6).mean(0)).tolist(),
                peak_requested_torque_Nm=np.abs(raw[:,js]).max(0).tolist(),
                peak_velocity_rad_s=np.abs(dq[:,js]).max(0).tolist(),
                hard_limit_occupancy_pct=(100*((q[:,js]<=hard[js,0]+.01)|(q[:,js]>=hard[js,1]-.01)).mean(0)).tolist(),
                target_outside_hard_limit_pct=(100*((target[:,js]<hard[js,0])|(target[:,js]>hard[js,1])).mean(0)).tolist(),
                target_foot_base_xyz_p95_p05_mm=((np.percentile(desired[:,i],95,axis=0)-np.percentile(desired[:,i],5,axis=0))*1000).tolist(),
                actual_foot_base_xyz_p95_p05_mm=((np.percentile(actual[:,i],95,axis=0)-np.percentile(actual[:,i],5,axis=0))*1000).tolist(),
                contact_point_speed_mean_m_s=float(slip[force>1].mean()))
            if air.any():row['noncontact_joint_mae_deg']=(np.abs(t20[air,js]-q20[air,js]).mean(0)*180/np.pi).tolist()
            legs.append(row)
        reports.append(dict(group=mode,trials=len(ids),valid_fraction=float(mask.mean()),
            failures=sum(r['index'] in ids for r in complete['resets']),
            vx_mean=float(d['velocity'][150:,ids][mask][:,0].mean()),legs=legs))
    pertrial=[]
    for i in range(18):
        v=valid[150:,i];g=d['gap'][150:,i][v]
        if len(g):pertrial.append(dict(index=i,gap_p95_mm=(np.percentile(g,95,axis=0)*1000).tolist(),mean_force_N=d['fz'][150:,i][v].mean(0).tolist()))
    report=dict(fk_validation_max_error_mm=fk_max_mm,statistics='Simulation 3-10s = forward 1-8s; 1ms joint telemetry, 20ms contacts',
        checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],cases=reports,pertrial=pertrial,
        limitations='Target FK is a hypothetical instantaneous geometry at target joint angles, not a separately commanded foot trajectory. Closed-loop mirrored policy intervention changes the full gait and body posture, not just one joint. Contact point speed is a rigid-body sphere-bottom estimate. Initial-state perturbations do not establish general robustness.')
    (run/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    time=np.arange(1,501)*.02-2;sel=(time>=2)&(time<=3.3)
    fig,axes=plt.subplots(4,3,figsize=(14,10),sharex=True,constrained_layout=True)
    for leg,name in enumerate(names):
        for joint in range(3):
            j=leg*3+joint;ax=axes[leg,joint]
            ax.plot(time[sel],d['target'][sel,0,j]*180/np.pi,label='Target',color='#d95f02')
            ax.plot(time[sel],d['q'][sel,0,j]*180/np.pi,label='Actual',color='#1b6ca8')
            ax.set_title(name+' / '+['Hip roll','Hip pitch','Knee'][joint]);ax.grid(alpha=.2)
            if joint==0:ax.set_ylabel('Angle (deg)')
            if leg==3:ax.set_xlabel('Forward time (s)')
    axes[0,0].legend();fig.suptitle('Normal model_500: target and actual joint angles | 0.35 m/s')
    fig.savefig(run/'target_vs_actual.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    colors=['#3478b9','#d62728','#309659','#9a60a7']
    for ax,idx,label in zip(axes,[0,9],['Normal policy','Diagnostic mirrored policy']):
        for i,name in enumerate(names):ax.plot(time[sel],d['gap'][sel,idx,i]*1000,label=name,color=colors[i])
        ax.set_ylabel('Clearance (mm)');ax.set_title(label);ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[1].set_xlabel('Forward time (s)')
    fig.savefig(run/'mirror_clearance.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(fk_max_mm=fk_max_mm,cases=reports),indent=2))

if __name__=='__main__':main()
