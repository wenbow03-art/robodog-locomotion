import json,os,subprocess,shutil,hashlib,math
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
run=Path((P/'latest_run.txt').read_text().strip());result=run/'result'
meta=json.loads((result/'metadata.json').read_text());trace=json.loads((result/'trace.json').read_text())
video=result/'soft_slip300_gait_lateral.mp4';ms=round(meta['physics_dt']*1000)
assert len(trace)==500 and meta['frames']==250 and meta['decimation']*meta['physics_dt']==.02
for k,source in meta['sources'].items():assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
for i,c in enumerate(([0.,.3,0.],[0.,-.3,0.])):
    assert all(r['segment']==i and r['command']==c and r['mirrored'] is False for r in trace[i*250:(i+1)*250])
ff=Path('/home/xiaohengli/miniconda3/envs/multinex/bin');env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
info=json.loads(subprocess.check_output([str(ff/'ffprobe'),'-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(video)],env=env))
assert abs(float(info['format']['duration'])-10)<.05 and int(info['streams'][0]['nb_frames'])==250
subprocess.run([str(ff/'ffmpeg'),'-v','error','-i',str(video),'-f','null','-'],check=True,env=env)
rows=[]
for i,seg in enumerate(meta['segments']):
    stop=seg['first_reset_step'] if seg['first_reset_step'] is not None else 250
    samples=trace[i*250+50:i*250+stop];row=dict(name=seg['name'],valid_samples=len(samples))
    if samples:
        mean=lambda xs:sum(xs)/len(xs)
        rolls=[]
        for s in samples:
            x,y,z,w=s['quaternion_xyzw'];rolls.append(abs(math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))))
        row.update(vx=mean([s['velocity'][0] for s in samples]),vy=mean([s['velocity'][1] for s in samples]),roll_abs_deg=mean(rolls),height_cm=100*mean([s['position'][2] for s in samples]),yaw_abs_deg_s=mean([abs(math.degrees(s['yaw_rate'])) for s in samples]),force_n=[mean([max(0,s['contact_force_z'][j]) for s in samples]) for j in range(4)])
    rows.append(row)
server=json.loads((ROOT/'media/v8_right_20261023/latest_result.json').read_text())['server']
leaf='soft_slip300_gait_%dms_'%ms+run.name;dest=Path(server['directory'])/leaf;dest.mkdir()
shutil.copyfile(video,dest/video.name)
subprocess.run([str(ff/'ffmpeg'),'-v','error','-ss','3','-i',str(video),'-frames:v','1',str(dest/'preview.jpg')],check=True,env=env)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V8侧移 {ms}ms</title><style>body{{max-width:1280px;margin:24px auto;background:#171a20;color:white;font:18px/1.6 system-ui}}video{{width:100%}}a{{color:#8cf}}</style><h1>触地滑动惩罚-0.05 · 侧移奖励3.0 · model300（左右均无镜像） · 物理步长{ms}ms · 策略50Hz</h1><p>左移0.30 m/s 5秒 → 右移0.30 m/s 5秒。左右均使用网络原生输出，不做镜像；连续切换，不做段间重置。开始前有2秒本次model300站立预热（不计入视频）。</p><video controls muted playsinline poster="preview.jpg" src="soft_slip300_gait_lateral.mp4"></video><p>重置记录：{len(meta['resets'])}。这是单次平地测试，本次用1ms物理步长、decimation=20；不是0.8秒周期训练模型。</p><a href="soft_slip300_gait_lateral.mp4" download>下载视频</a></html>'''
(dest/'index.html').write_text(page)
report=dict(url='http://localhost:%d/%s/'%(server['port'],leaf),video=str(video),metadata=meta,analysis=rows,video_check=info)
for target in [P/'latest_result.json',run/'result.json']:
    target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(url=report['url'],rows=rows,resets=meta['resets']),indent=2))
