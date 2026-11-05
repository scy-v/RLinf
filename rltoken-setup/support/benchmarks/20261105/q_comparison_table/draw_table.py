"""Render the three-experiment local Q comparison as a presentation table."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
O=Path(__file__).resolve().parent
regular=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
bold=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['pdf.fonttype']=42
fig,ax=plt.subplots(figsize=(12,3.82))
fig.patch.set_facecolor('white');ax.set(xlim=(.35,11.65),ylim=(1.24,4.97));ax.axis('off')
navy='#243454';purple='#6047A7';line='#D9DEEA'
xs=[.45,2.8,4.65,6.55,8.3,11.55]
headers=['更新模式','UTD','Critic∶Actor','控制轴','状态ΔQ / 动作ΔQ']
rows=[['额度更新','5','2∶1','RP锁轴','约 21–66 倍'],['额度更新','10','4∶1','全轴控制','约 45–198 倍'],['持续更新','不设额度','4∶1','RP锁轴','约 1187–2042 倍']]
ytop=4.87;hh=.8;rh=.91
for j,h in enumerate(headers):
 ax.add_patch(Rectangle((xs[j],ytop-hh),xs[j+1]-xs[j],hh,facecolor=navy if j<4 else purple,edgecolor='white',linewidth=1))
 ax.text((xs[j]+xs[j+1])/2,ytop-hh/2,h,fontproperties=bold,fontsize=16 if j<4 else 15,color='white',ha='center',va='center')
for i,row in enumerate(rows):
 y=ytop-hh-(i+1)*rh
 for j,val in enumerate(row):
  bg=('#F0EBFA' if i%2==0 else '#E8E0F5') if j==4 else ('#FFFFFF' if i%2==0 else '#F3F5FA')
  ax.add_patch(Rectangle((xs[j],y),xs[j+1]-xs[j],rh,facecolor=bg,edgecolor=line,linewidth=.8))
  ax.text((xs[j]+xs[j+1])/2,y+rh/2,val,fontproperties=bold if j==4 else regular,fontsize=18 if j==4 else 17,color=purple if j==4 else navy,ha='center',va='center')
ax.plot([.45,11.55],[1.34,1.34],color=purple,lw=2)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for ext in ['png','pdf']:
 fig.savefig(O/f'q_state_action_comparison.{ext}',dpi=220,facecolor='white')
plt.close(fig)
