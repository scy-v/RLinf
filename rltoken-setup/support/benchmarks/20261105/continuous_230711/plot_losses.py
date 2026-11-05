"""Plot fixed 250-runner-round averages; preserve raw metrics in source logs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
O=Path(__file__).resolve().parent
R=Path('logs/online_rltoken/20261104-230711-ur_rlt_stage2_ac_mlp')
rows=[json.loads(l) for l in (R/'metrics.log').read_text().splitlines() if l.strip()]
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':12,'axes.unicode_minus':False})
bins={}
for r in rows:bins.setdefault(r['step']//250,[]).append(r)
t0=rows[0]['start_time']
x=[np.mean([r['timestamp']-t0 for r in rs])/60 for rs in bins.values()]
fig,axs=plt.subplots(2,2,figsize=(11,7.5))
for ax,keys,title in zip(axs.flat,[['train/actor/q_pi','train/critic/q_data'],['train/actor/bc_loss','train/actor/bc_human_loss','train/actor/bc_ref_loss'],['train/sac/critic_loss'],['train/sac/actor_loss']],['平均Q','BC误差','Critic TD误差','Actor总损失']):
 for k in keys:
  ax.plot(x,[np.mean([r['metrics'][k] for r in rs]) for rs in bins.values()],label=k.split('/')[-1],lw=2)
 ax.axvspan(30.1,43.2,alpha=.08,color='green',label='第55–78轮阶段')
 ax.set(title=title,xlabel='训练时间（分钟）');ax.grid(alpha=.2);ax.legend(fontsize=9)
 if title in ['BC误差','Critic TD误差']:ax.set_yscale('log')
fig.tight_layout();fig.savefig(O/'loss_curves.png',dpi=170)
