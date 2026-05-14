"""檢查哪張圖在 Label.txt 裡漏掉沒標"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

labeled = set()
for root in [Path('C:/Users/andy_ac_chen/Desktop/backup/success'),
             Path('C:/Users/andy_ac_chen/Desktop/backup/fail')]:
    for lf in root.rglob('Label.txt'):
        for line in lf.read_text(encoding='utf-8').splitlines():
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            try:
                json.loads(parts[1])
            except:
                continue
            rel = parts[0].replace('\\', '/')
            fname = Path(rel).name
            labeled.add((lf.parent.parent.name, lf.parent.name, fname))

actual = set()
for root in [Path('C:/Users/andy_ac_chen/Desktop/backup/success'),
             Path('C:/Users/andy_ac_chen/Desktop/backup/fail')]:
    for jpg in root.rglob('*.jpg'):
        actual.add((jpg.parent.parent.name, jpg.parent.name, jpg.name))

print(f'Label.txt 標到: {len(labeled)}')
print(f'實際 jpg:       {len(actual)}')
print(f'差距:           {len(actual) - len(labeled)}')

missing = actual - labeled
print(f'\n漏標的 {len(missing)} 張:')
for m in sorted(missing):
    print(f'  {m[0]}/{m[1]}/{m[2]}')

extra = labeled - actual
if extra:
    print(f'\nLabel.txt 有但圖檔不存在的 {len(extra)} 張:')
    for m in sorted(extra):
        print(f'  {m[0]}/{m[1]}/{m[2]}')
