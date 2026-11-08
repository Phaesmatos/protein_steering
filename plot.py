import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('metrics')
p.add_argument('--out', default=None)
a = p.parse_args()
with open(a.metrics, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('No data')
metric = 'mean_p_g' if 'mean_p_g' in rows[0] else 'mean_p_native_aa'
series_key = 'phase' if 'phase' in rows[0] else 'group'
groups = sorted({(r['condition'], r[series_key]) for r in rows})
for condition, group in groups:
    points = sorted(((float(r['alpha']), float(r[metric])) for r in rows
                     if r['condition'] == condition and r[series_key] == group))
    plt.plot([x for x, _ in points], [y for _, y in points], marker='o', label=f'{condition} / {group}')
plt.xlabel('Steering coefficient alpha (unit-norm direction)')
plt.ylabel(metric)
plt.ylim(0, 1)
plt.title('ESM-2 activation steering: in-silico diagnostic')
plt.legend(fontsize=8)
plt.tight_layout()
out = Path(a.out or Path(a.metrics).with_suffix('.png'))
plt.savefig(out, dpi=180)
print(f'Saved {out}')
