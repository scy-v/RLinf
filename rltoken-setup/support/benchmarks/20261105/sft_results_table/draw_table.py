"""Render the first core-results table from rltoken-setup/summary.md."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

out = Path(__file__).resolve().parent
source = out.parents[3] / 'summary.md'
lines = source.read_text().split('## 核心结果', 1)[1].split('## 结论', 1)[0].splitlines()
rows = [[c.strip() for c in line.strip('|').split('|')] for line in lines if line.startswith('|')]
headers, data = rows[0], rows[2:]
assert len(data) == 6 and len(headers) == 5
regular = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=19)
bold = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size=19)
plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(figsize=(12, 5.7))
ax.axis('off')
table = ax.table(cellText=data, colLabels=headers, cellLoc='center', colWidths=[.18,.20,.20,.20,.22], bbox=[.005,.01,.99,.98])
table.auto_set_font_size(False)
for (r,c), cell in table.get_celld().items():
    cell.set_edgecolor('#DED8D5')
    cell.set_linewidth(.8)
    cell.get_text().set_fontproperties(bold if r == 0 or c == 0 else regular)
    if r == 0:
        cell.set_facecolor('#723F43')
        cell.get_text().set_color('white')
    else:
        cell.set_facecolor('#F7E9E5' if c == 0 else ('#FAF6F3' if r % 2 == 0 else 'white'))
        cell.get_text().set_color('#703E42' if c == 0 else '#343139')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
for ext in ['png','pdf']:
    fig.savefig(out / f'sft_results_table.{ext}', dpi=220, facecolor='white')
plt.close(fig)
