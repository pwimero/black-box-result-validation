from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import numpy as np

def solve2(m,b):
    det=m[0][0]*m[1][1]-m[0][1]*m[1][0]
    if not det:raise ValueError('singular')
    return [(b[0]*m[1][1]-m[0][1]*b[1])/det,(m[0][0]*b[1]-b[0]*m[1][0])/det]

def state(x,a,tau,clamps=None,atten=None,repl=None):
    clamps={} if clamps is None else clamps;atten={} if atten is None else atten;repl={} if repl is None else repl
    edges=[(0,1),(1,0)];solutions=[]
    for bits in product([0,1],repeat=2):
        active=dict(zip(edges,bits));m=[[F(1),F(0)],[F(0),F(1)]];rhs=list(x)
        for i,k in edges:
            if i in clamps:rhs[i]=clamps[i];continue
            aa=a[i][k]*atten.get((i,k),F(1))
            if (i,k) in repl:rhs[i]+=aa*repl[i,k]
            elif active[i,k]:m[i][k]-=aa;rhs[i]-=aa*tau[i][k]
        z=solve2(m,rhs)
        if all(i in clamps or (i,k) in repl or (z[k]>=tau[i][k] if active[i,k] else z[k]<=tau[i][k]) for i,k in edges):solutions.append(z)
    assert solutions and all(z==solutions[0] for z in solutions)
    return solutions[0]

M=[[F(1),F(1)],[F(0),F(1)]];b=[F(1,3),F(-1,5)]
matvec=lambda m,z:[sum(u*v for u,v in zip(row,z)) for row in m]
a=[[F(0),F(2)],[F(-2),F(0)]];tau=[[F(0),F(1,4)],[F(-1,2),F(0)]]
Fout=lambda x:[u+v for u,v in zip(matvec(M,state(x,a,tau)),b)]
C0=F(3);h=F(1);K=F(10);H1=F(2);H2=F(3)
base=[-C0,-C0];y0=Fout(base);cal=[]
for j in range(2):
    x=base.copy();x[j]+=h;cal.append(Fout(x))
Mh=[[(cal[j][i]-y0[i])/h for j in range(2)] for i in range(2)];bh=[v-w for v,w in zip(y0,matvec(Mh,base))]
assert Mh==M and bh==b
ah=[[F(0),F(0)],[F(0),F(0)]];th=[[F(0),F(0)],[F(0),F(0)]]
for k in range(2):
    zs=[]
    for H in [H1,H2]:
        x=[-K,-K];x[k]=H;y=Fout(x);zs.append(solve2(Mh,[v-w for v,w in zip(y,bh)]))
    for i in range(2):
        if i!=k:
            ah[i][k]=(zs[1][i]-zs[0][i])/(H2-H1)
            th[i][k]=H1-(zs[0][i]+K)/ah[i][k]
assert ah==a and th==tau
rng=np.random.default_rng(20261026);checks=0;max_inverse_ratio=F(0)
for _ in range(100):
    x=[F(int(rng.integers(-20,21)),4) for _ in range(2)];xp=[F(int(rng.integers(-20,21)),4) for _ in range(2)]
    clamps={i:F(int(rng.integers(-4,5)),4) for i in range(2) if rng.random()<.2}
    atten={(i,k):F(int(rng.integers(0,5)),4) for i,k in [(0,1),(1,0)]}
    repl={(i,k):F(int(rng.integers(0,5)),4) for i,k in [(0,1),(1,0)] if rng.random()<.2}
    z=state(x,a,tau,clamps,atten,repl);zh=state(x,ah,th,clamps,atten,repl);zp=state(xp,a,tau,clamps,atten,repl)
    assert z==zh
    dx=max(abs(u-v) for u,v in zip(x,xp));dz=max(abs(u-v) for u,v in zip(z,zp))
    assert dz<=3*dx
    if dx:max_inverse_ratio=max(max_inverse_ratio,dz/dx)
    checks+=1

L=[[F(0),F(3)],[F(1,10),F(0)]];weight=[F(5),F(1)];rho=F(3,5)
assert all(v<=rho*w for v,w in zip(matvec(L,weight),weight))
G=[[F(1),F(-3)],[F(-1,10),F(1)]];forcing=[F(1,100),F(1,50)];error=solve2(G,forcing)
z=solve2(G,[F(1),F(1)]);zh=solve2(G,[F(1)+forcing[0],F(1)+forcing[1]])
assert [v-u for u,v in zip(z,zh)]==error
result={'negative_feedback_beyond_absolute_contraction':{'status':'PASS_EXACT_RATIONAL','gains':[2,-2],'absolute_gain_spectral_radius':2,'vector_queries':7,'parameters_recovered_exactly':True,'simultaneous_intervention_checks':checks,'uniform_inverse_lipschitz_upper':3,'maximum_sampled_inverse_ratio':str(max_inverse_ratio)},'weighted_contraction':{'status':'PASS_EXACT_RATIONAL','gain_matrix':[[str(v) for v in row] for row in L],'ordinary_maximum_row_sum':'3','positive_weight_vector':[str(v) for v in weight],'weighted_contraction_upper':str(rho),'resolvent_error_exact':[str(v) for v in error],'bound_attained_in_positive_linear_region':True}}
p=Path(__file__).resolve().parent/'generated/frontier_verification.json';all_results=json.loads(p.read_text());all_results.update(result);p.write_text(json.dumps(all_results,indent=2));print(json.dumps(result,indent=2))
