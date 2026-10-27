"""Exact integer-normalized replay of the unchanged historical finite-table checker.

All rationals are placed over one common denominator. This changes arithmetic
cost only: |claim gap|-L*max|response gap| has numerator
L.denominator*|claim integer gap|-L.numerator*max|response integer gap|.
The first maximum witness is retained, matching the reference enumeration.
"""
from fractions import Fraction as F
from math import lcm
from pathlib import Path
import importlib.util,json


def stable_error(claims,responses,lipschitz):
    if len(claims)!=len(responses): raise ValueError('shape')
    dimensions={len(row) for row in responses}
    if len(dimensions)!=1 or not next(iter(dimensions)): raise ValueError('response shape')
    scale=lcm(*(v.denominator for v in claims),*(v.denominator for row in responses for v in row))
    a=[v.numerator*(scale//v.denominator) for v in claims]
    b=[tuple(v.numerator*(scale//v.denominator) for v in row) for row in responses]
    best=0; witness=(0,0); p=lipschitz.numerator; q=lipschitz.denominator
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            defect=q*abs(a[i]-a[j])-p*max(abs(x-y) for x,y in zip(b[i],b[j],strict=True))
            if defect>best: best=defect; witness=(i,j)
    return F(best,2*scale*q),witness


def load_checker(path):
    spec=importlib.util.spec_from_file_location('historical_finite_checker',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    legacy=root/'legacy' if (root/'legacy').exists() else root
    checker=load_checker(legacy/'reference_checker/check_jmlr_prospective_suite.py')
    checker.stable_error=stable_error
    result=checker.replay(legacy/'results/jmlr_prospective_suite.json',legacy/'results/jmlr_prospective_world_tables.json.gz',legacy)
    result['arithmetic']='EXACT_COMMON_DENOMINATOR_INTEGERS'
    result['scope']='Unchanged historical validation logic with algebraically equivalent integer pair arithmetic'
    (root/'reports/prospective_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
