"""Check the complete recording and serve it through the existing local video page."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xiaohengli/Program/robo_competition')
FF = Path('/home/xiaohengli/miniconda3/envs/multinex/bin')
folder = ROOT/'media/v3_stand_walk_20261027'
run = Path((folder/'latest_run.txt').read_text().strip())
result = run/'result'
meta = json.loads((result/'metadata.json').read_text())
trace = json.loads((result/'trace.json').read_text())
video = result/'hybrid_stand_walk_transition.mp4'
assert meta['iteration'] == 800 and meta['frames'] == 800 and len(trace) == 1600
assert len(meta['segments']) == 4
assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest() == meta['checkpoint_sha256']
for source, digest in meta['source_sha256'].items():
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
for i, command in enumerate(([0., 0., 0.], [.35, 0., 0.], [0., 0., 0.], [.35, 0., 0.])):
    assert all(row['segment'] == i and row['command'] == command and not row['mirrored']
               for row in trace[i*400:(i+1)*400])
env = os.environ.copy()
env.pop('LD_LIBRARY_PATH', None)
info = json.loads(subprocess.check_output([str(FF/'ffprobe'), '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration,size',
    '-of', 'json', str(video)], env=env))
stream = info['streams'][0]
assert stream['codec_name'] == 'h264' and int(stream['nb_frames']) == 800
assert (stream['width'], stream['height']) == (1280, 480)
assert abs(float(info['format']['duration']) - 32) < .05
subprocess.run([str(FF/'ffmpeg'), '-v', 'error', '-i', str(video), '-f', 'null', '-'], env=env, check=True)
public = run/'public'
public.mkdir(exist_ok=False)
shutil.copyfile(video, public/video.name)
for i in range(4):
    subprocess.run([str(FF/'ffmpeg'), '-v', 'error', '-ss', str(i*8+4), '-i', str(video),
        '-frames:v', '1', str(public/('segment_%d.jpg' % i))], env=env, check=True)
resets = sum(r['phase'] == 'video' for r in meta['resets'])
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baseline站立 / model800直行 · 连续切换</title>
<style>body{max-width:1280px;margin:28px auto;padding:0 18px;background:#151820;color:#edf0f6;font:17px/1.6 system-ui}video{width:100%;background:#000;border-radius:8px}button{padding:10px 16px;margin:4px;background:#32465f;color:white;border:1px solid #587292;border-radius:6px;cursor:pointer}a{color:#8ecbff}</style>
<h1>Baseline站立 / model800直行</h1><p>站立、直行、停下、再次起步，各8秒。直行指令：0.35 米/秒。原速播放。</p>
<video id="v" controls muted playsinline preload="metadata" poster="segment_0.jpg" src="hybrid_stand_walk_transition.mp4"></video>
<p><button onclick="jump(0)">站立 · 0–8 秒</button><button onclick="jump(8)">直行 · 8–16 秒</button><button onclick="jump(16)">停下 · 16–24 秒</button><button onclick="jump(24)">再起步 · 24–32 秒</button></p>
<p>左画面：右侧视角；右画面：右后方视角。全程连续，切换指令时不重置。右后腿修正在0.3秒内加入或撤掉。画面内自动重置 RESET_COUNT 次。</p>
<p>单次平地仿真演示，未开启随机扰动。</p>
<p><a href="hybrid_stand_walk_transition.mp4" download>下载 MP4</a></p>
<script>function jump(t){const v=document.getElementById('v');v.currentTime=t;v.play().catch(()=>{});}</script></html>'''.replace('RESET_COUNT', str(resets))
(public/'index.html').write_text(page)
server = json.loads((ROOT/'media/v8_right_20261023/latest_result.json').read_text())['server']
leaf = 'new_v3_stand_walk_' + run.name
destination = Path(server['directory'])/leaf
shutil.copytree(public, destination)
published = dict(url='http://localhost:%d/%s/' % (server['port'], leaf), video=str(video),
    server_directory=str(destination), metadata=meta, video_check=info)
for path in (run/'result.json', folder/'latest_result.json'):
    path.write_text(json.dumps(published, indent=2)+'\n')
print(json.dumps(dict(url=published['url'], video=str(video), segments=meta['segments'], resets=meta['resets']), indent=2))
