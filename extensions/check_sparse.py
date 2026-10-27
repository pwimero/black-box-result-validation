"""Observation-only rational checker for m=2, s=1. No ground truth or producer imports."""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inverse(a):
    n = len(a)
    b = [list(row) + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        k = next((k for k in range(j, n) if b[k][j]), None)
        require(k is not None, 'singular matrix')
        b[j], b[k] = b[k], b[j]
        pivot = b[j][j]
        b[j] = [v / pivot for v in b[j]]
        for i in range(n):
            if i != j:
                q = b[i][j]
                b[i] = [v - q * w for v, w in zip(b[i], b[j])]
    return [row[n:] for row in b]


def mv(a, x):
    return [sum((v * w for v, w in zip(row, x)), F(0)) for row in a]


def norm(x):
    return max(map(abs, x), default=F(0))


def matrix_norm(a):
    return max(sum(map(abs, row)) for row in a)


def check(record):
    require(record['schema'] == 'tcabb.sparse.v1', 'schema')
    d = record['d']; m = record['m']
    require(isinstance(d, int) and d >= 2 and m == 2 and record['s'] == 1, 'dimensions')
    p = {k: F(v) for k, v in record['bounds'].items()}
    A, amin, T, R, J, H1, H2, K, h, C0, eta = [p[k] for k in
        ['A','amin','T','R','J','H1','H2','K','h','C0','eta']]
    require(A >= amin > 0 and min(T,R,J,eta) >= 0 and h > 0, 'bounds')
    require(H2 > H1 > T and K > max(H2,T+A*(H2+T)) and C0 > T+h, 'geometry')
    obs = [[F(v) for v in row] for row in record['observations']]
    require(len(obs) == 3*d+1 and all(len(row) == m for row in obs), 'transcript shape')
    C = [[(obs[j+1][i]-obs[0][i])/h for j in range(d)] for i in range(m)]
    b = [obs[0][i] + C0 * sum(C[i]) for i in range(m)]
    margins = []
    for j,k in combinations(range(d),2):
        inv = inverse([[row[j],row[k]] for row in C])
        margins.append(1/matrix_norm(inv))
    kap = min(margins)
    em = 2*d*eta/h; eb = eta+C0*em
    eu = 2*eta/(H2-H1)+em*max(1,A)
    ev = 2*H1*eta/(H2-H1)+eta+eb+em*(K+A*T)
    ea = 2*eu/kap; et = (2*ev/kap+T*ea)/amin
    require(ea < amin, 'support margin')
    gain = [[F(v) for v in row] for row in record['gain']]
    prod = [[F(v) for v in row] for row in record['product']]
    require(len(gain)==d and len(prod)==d and all(len(r)==d for r in gain+prod), 'fit shape')
    for k in range(d):
        a = [row[k] for row in gain]; v = [row[k] for row in prod]
        require(a[k] == v[k] == 0, 'nonzero diagonal')
        require(sum(bool(t) for t in a) <= 1, 'sparsity')
        require(all(t == 0 or amin <= abs(t) <= A for t in a), 'gain interval')
        require(all(not v[i] or a[i] for i in range(d)), 'product support')
        y1,y2 = obs[d+1+2*k:d+3+2*k]
        ell = [(y2[i]-y1[i])/(H2-H1) for i in range(m)]
        u = [ell[i]-C[i][k] for i in range(m)]
        w = [H1*ell[i]-y1[i]-K*sum(C[i])+K*C[i][k]+b[i] for i in range(m)]
        require(norm([x-y for x,y in zip(mv(C,a),u)]) <= eu, 'gain residual')
        require(norm([x-y for x,y in zip(mv(C,v),w)]) <= ev, 'product residual')
    tau = [[max(-T,min(T,prod[i][k]/gain[i][k])) if gain[i][k] else F(0)
            for k in range(d)] for i in range(d)]
    U = [[min(A,abs(t)+ea) if t else F(0) for t in row] for row in gain]
    weight = list(map(F,record['weight']))
    require(len(weight)==d and all(t>0 for t in weight), 'positive weight')
    require(all(t<w for t,w in zip(mv(U,weight),weight)), 'contraction')
    resolvent = inverse([[F(i==j)-U[i][j] for j in range(d)] for i in range(d)])
    B = mv(resolvent,[max(R,J)+T*sum(row) for row in U])
    force = [sum(ea*(B[k]+T)+abs(gain[i][k])*et for k in range(d) if gain[i][k]) for i in range(d)]
    D = mv(resolvent,force)
    radius = em*norm(B)+matrix_norm(C)*norm(D)+eb
    return dict(status='ACCEPT', arithmetic='EXACT_RATIONAL', C=C,b=b,tau=tau,
                kappa=kap, gain_error=ea, threshold_error=et, radius=radius,
                state_bound=B,state_error=D,queries=len(obs))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('certificate',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(json.loads(args.certificate.read_text())),default=str,indent=2))
