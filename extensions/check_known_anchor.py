"""Exact gradient verification, conditional on independently replayed PWA coverage/continuity."""
from fractions import Fraction as F
import gzip,json,hashlib
from pathlib import Path

EXPECTED='077ef1831a58db8f0cdf3d0ffe59d01cc23c6762f57b52c69913916254bb5113'
RATIOS=[F(3064460033040940540216,1676191798982764748491),
        F(1975176615853348368583,925306615629314381572),
        F(77820881779556097080,14074347043342301049),
        F(942443824311307814971,625837459604625310495)]


def gradient(affine):
    c,p,q=[F(int(n),int(d)) for n,d in affine]
    return p-q,q


def check(path):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise ValueError('sidecar hash mismatch')
    data=json.loads(gzip.decompress(raw)); records=[]
    for h,cells,expected in zip([8,9,10,11],data['anchors'],RATIOS,strict=True):
        C=max(sum(map(abs,gradient(cell['claim']))) for cell in cells)
        margins=[]
        for cell in cells:
            gu,gv=gradient(cell['hard_responses'][3])
            hu,hv=gradient(cell['hard_responses'][1])
            margins.extend([gu-abs(gv),-hv-abs(hu)])
        m=min(margins)
        if not m>0 or C/m != expected or C/m>6: raise ValueError('gradient certificate failed')
        upper=F(-(-expected.numerator*10000//expected.denominator),10000)
        records.append(dict(anchor=h,cells=len(cells),claim_gradient_bound=C,response_margin=m,
                            ratio=C/m,rounded_upper=upper))
    return dict(status='PASS_EXACT_RATIONAL',sidecar_sha256=EXPECTED,
                scope='Requires continuous partition coverage checked by continuous_replay.json',anchors=records)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    path=root/'legacy/results/continuous_triangle_duals.json.gz'
    if not path.exists(): path=root/'results/continuous_triangle_duals.json.gz'
    report=check(path)
    (root/'reports/known_anchor.json').write_text(json.dumps(report,default=str,indent=2)+'\n')
    print(json.dumps(report,default=str,indent=2))
