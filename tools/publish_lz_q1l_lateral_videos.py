"""Validate recordings and reuse the existing localhost video service."""
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/xiaohengli/Program/robo_competition')


def head(url, expected_size):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=5) as response:
        assert response.status == 200
        assert int(response.headers['Content-Length']) == expected_size


def main():
    if sys.argv[1] == '--serve':
        public, port, ready = Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(public))
        server = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
        ready.write_text(json.dumps(dict(pid=os.getpid(), port=port, directory=str(public))))
        server.serve_forever()
        return
    run = Path(sys.argv[1]).resolve()
    public = run / 'public'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert hashlib.sha256(Path(manifest['checkpoint']).read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    for name, digest in manifest['source_sha256s'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    metadata = {}
    cards = []
    for direction, title, branch, sign in [('left', '左移', '镜像策略', 1), ('right', '右移', '原策略', -1)]:
        info = json.loads((run / (direction + '.check.json')).read_text())
        assert abs(float(info['format']['duration']) - 12.) < .05
        assert int(info['streams'][0]['nb_frames']) == 300
        meta = json.loads((public / (direction + '.json')).read_text())
        assert meta['policy_mode'] == 'left-mirror'
        assert meta['schedule'][1] == [8., [0., sign * .4, 0.]]
        trace = json.loads((public / (direction + '.trace.json')).read_text())
        moving = [r for r in trace if r['command'][1] != 0.]
        assert len(trace) == 600 and len(moving) == 400
        assert all(r['mirrored'] == (direction == 'left') for r in moving)
        metadata[direction] = meta
        speed = meta['mean_vy_after_1s_before_first_reset']
        speed_text = '%+.3f m/s' % speed if speed is not None else '无有效样本'
        cards.append('''<section><h2>%s · %s</h2>
<video id="%s" controls muted playsinline preload="metadata" poster="%s.jpg" src="%s.mp4"></video>
<p>目标 vy：%+.1f m/s　｜　侧移段平均 vy：%s　｜　重置：%d 次</p>
<p class="status" id="%s-status">0–2 秒站立 · 2–10 秒侧移 · 10–12 秒站立</p>
<button onclick="jump('%s')">从侧移开始播放</button> <a href="%s.mp4" download>下载%s MP4</a>
</section>''' % (title, branch, direction, direction, direction, sign * .4,
                 speed_text, len(meta['resets']), direction, direction, direction, title))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V8 左移与右移仿真视频</title>
<style>body{max-width:1320px;margin:30px auto;padding:0 18px;background:#151820;color:#edf0f6;font:16px/1.6 system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}section{background:#202733;padding:18px;border-radius:12px}h2{margin-top:0}video{width:100%;border-radius:8px;background:#000}a{color:#8ecbff}button{padding:8px 12px;color:#edf0f6;background:#32465f;border:1px solid #587292;border-radius:6px;cursor:pointer}.status{font-variant-numeric:tabular-nums;min-height:3em}</style>
<h1>V8 · 左移与右移仿真视频</h1>
<p>左移自动采用镜像策略，右移采用原策略。方向以机器人自身朝向为准。</p>
<main>''' + '\n'.join(cards) + '''</main>
<p>两段均使用原 V8 model200、相同随机种子及默认组合控制器。每段 12 秒，侧移指令持续 8 秒。
平均速度排除侧移起步第一秒和首次重置后的样本。这两段录像仅展示本次运行。</p>
<script>
function jump(id){const v=document.getElementById(id);v.currentTime=2;v.play();}
for(const id of ['left','right']){
 const v=document.getElementById(id),status=document.getElementById(id+'-status');let trace=[];
 fetch(id+'.trace.json').then(r=>r.json()).then(d=>trace=d);
 v.addEventListener('timeupdate',()=>{const r=trace[Math.min(trace.length-1,Math.floor(v.currentTime/.02))];
 if(r)status.textContent=v.currentTime.toFixed(1)+' 秒 · 指令 vy '+r.command[1].toFixed(2)+' m/s · 实测 vy '+r.velocity[1].toFixed(3)+' m/s · '+(r.mirrored?'镜像策略':'原策略');});
}
</script></html>'''
    (public / 'index.html').write_text(page)
    previous = ROOT / 'media/v8_right_20261023/latest_result.json'
    server = None
    if previous.exists():
        old = json.loads(previous.read_text())
        try:
            head(old['url'] + Path(old['video']).name, Path(old['video']).stat().st_size)
        except (OSError, AssertionError):
            pass
        else:
            server = old['server']
            leaf = 'lateral_' + run.name
            destination = Path(server['directory']) / leaf
            shutil.copytree(public, destination)
            url = old['url'] + leaf + '/'
    if server is None:
        log = (run / 'http.log').open('a')
        for port in range(8769, 8779):
            ready = run / ('server_%d.ready.json' % port)
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve',
                                      str(public), str(port), str(ready)], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
            for _ in range(40):
                if ready.exists() or child.poll() is not None:
                    break
                time.sleep(.1)
            if ready.exists():
                server = json.loads(ready.read_text())
                url = 'http://127.0.0.1:%d/' % port
                break
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
        if server is None:
            raise RuntimeError('No available localhost video port')
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200 and '左移与右移'.encode() in response.read()
    for direction in metadata:
        head(url + direction + '.mp4', (public / (direction + '.mp4')).stat().st_size)
    result = dict(state='recorded_and_serving', server=server,
                  url=url.replace('127.0.0.1', 'localhost'), recordings=metadata,
                  local_public=str(public))
    (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (run.parent / 'latest_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
