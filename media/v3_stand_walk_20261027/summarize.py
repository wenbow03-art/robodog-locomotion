from pathlib import Path
import json,hashlib,shutil
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];d=json.loads((P/'evaluation/report.json').read_text());published=json.loads((P/'latest_result.json').read_text());r=Path(published['video']).parent;trace=json.loads((r/'trace.json').read_text())
alpha=np.array([x['alpha'] for x in trace]);assert len(alpha)==1600
for i,goal in enumerate([0,1,0,1]):assert np.all(alpha[i*400+15:(i+1)*400]==goal)
assert np.max(abs(np.diff(alpha)))<=1/15+1e-6
manifest=json.loads((P/'implementation_manifest.json').read_text())
for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
assert not d['resets'];assert not published['metadata']['resets']
g={(x['controller'],x['stage']):x for x in d['groups']}
lines=['# Baseline站立 / model800直行：连续切换结果','', '已实现推理任务 lz_q1l_flat_stand_walk。0.30秒（15个控制步）加入/撤销右后腿修正；不训练、不改权重或奖励。保持正常左侧移镜像控制，未对侧移/转弯做新效果保证。','', '## 实现及检查','', '- 配置：lz_q1l_2_config.py 的 LZQ1LStandWalkCfg.inference。', '- 环境工厂与逐环境复位：lz_q1l_2.py 的 LZQ1LStandWalkRobot。', '- 修正分支插值：utils/lz_q1l_stand_walk.py 的 StandWalkResidualActor。', '- 站立端点输出逐元素等于冻结baseline；直行端点逐元素等于model800；其他9关节输出完全不受插值修改（相同输入下）。检查包括部分环境复位、过渡中途反向、左侧移镜像及参数不变。', '- 过渡是右后修正分支的权重平滑，不代表全部关节指令在命令突变时不发生变化。', '', '## 回放','', '[连续32秒视频]('+published['url']+')','', '视频：初始仅重置一次、预站立2秒；站立8秒→直行8秒→停车8秒→再次直行8秒，段间不重置，未剪除失败。另做baseline/model800/混合策略各3个初始根速度状态，共9条完整32秒回放、27次指令切换，无跌倒重置。下面统计各段最后5秒；3个初态不是独立训练种子。','','|控制器|阶段|侧倾°|后腿力差N|前腿力差N|左侧承重%|高度cm|','|---|---|---:|---:|---:|---:|---:|']
for controller in ['baseline','model800','hybrid']:
 for stage in ['stand','walk','stop','restart']:
  x=g[controller,stage];lines.append('|%s|%s|%s|'%(controller,stage,'|'.join('%.2f'%x[k]['mean'] for k in ['roll_abs_deg','rear_gap_n','front_gap_n','left_share_pct','height_cm'])))
lines+=['','## 解释','', '1. 切换功能正确，0.3秒插值符合配置，初始站立使用baseline，行走/再次起步保留model800。混合策略第一次行走后/前腿力差1.20/0.79N，左侧49.92%；再次起步后0.57/0.48N，左侧50.03%。','2. 停下后身体高度恢复约29.12cm、四脚100%触地、侧倾仅0.34度，但承重明显偏置：LH75.18/RH14.67/LF21.88/RF38.01N，左侧64.82%，后腿力差60.51N。这是停车后最后5秒的持续现象，不只是0.3秒切换瞬间。','3. 对照的纯baseline走完再停同样出现LH80.23/RH11.28N、后腿差68.95N、左侧67.58%。因此该问题并非只在混合策略出现。证据支持baseline本身尚不能从本次行走结束状态恢复到初始站立受力；脚落点、姿态和历史状态与初始化不同是可能原因，尚未单独分离这些因素。','4. 初始站立的三个控制器处于不同环境位置/回放数值轨迹，静态力分配有初态敏感性；不能把“相同输入时输出相同”解释为闭环轨迹和受力永远相同。','5. 此实现可以作为切换演示，但没有解决停车后的承重恢复。后续应单独研究减速停车/落脚与站姿恢复，不能通过在停车段重置仿真来假装问题消失。本次未做进一步调参或训练。','']
(P/'findings_zh.md').write_text('\n'.join(lines)+'\n')
public=Path(published['server_directory']);shutil.copyfile(P/'findings_zh.md',public/'findings_zh.md');shutil.copyfile(P/'evaluation/report.json',public/'comparison.json')
for page in [public/'index.html',Path(published['video']).parents[1]/'public/index.html']:
 s=page.read_text();s=s.replace('<script>', '<p><strong>连续停车测试：无跌倒重置，但停车后仍存在明显承重偏置；并非已经解决站立平衡。</strong> <a href="findings_zh.md">查看完整对比报告</a></p><script>');page.write_text(s)
print(json.dumps(dict(url=published['url'],resets=d['resets'],alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),max_alpha_step=float(abs(np.diff(alpha)).max())),indent=2))
