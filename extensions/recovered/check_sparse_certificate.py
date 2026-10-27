#!/usr/bin/env python3
"""Observation-only exact checker for two-output, one-sparse ReLU certificates.

No producer, neural model, ground truth, numerical array package, or optimizer
is imported. The checker validates a conditional certificate under the declared
structural class, probe geometry, labelled actuation, and bounded-noise contract.
It does not establish those premises for an unrestricted physical black box.
"""
import argparse
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path


def require(test,message):
    if not test:raise ValueError(message)


def vec(raw,n):
    require(isinstance(raw,list) and len(raw)==n,'vector dimension mismatch')
    return [F(x) for x in raw]


def matrix(raw,n,m):
    require(isinstance(raw,list) and len(raw)==n,'matrix dimension mismatch')
    return [vec(row,m) for row in raw]


def norm(v):return max(map(abs,v),default=F(0))
def mnorm(a):return max(sum(map(abs,row)) for row in a)
def mv(a,v):return [sum(x*y for x,y in zip(row,v)) for row in a]
def subtract(a,b):return [x-y for x,y in zip(a,b)]


def solve(a,b):
    n=len(a);rows=[list(row)+[b[i]] for i,row in enumerate(a)]
    for k in range(n):
        pivot=next((i for i in range(k,n) if rows[i][k]),None)
        require(pivot is not None,'singular resolvent')
        rows[k],rows[pivot]=rows[pivot],rows[k]
        scale=rows[k][k];rows[k]=[x/scale for x in rows[k]]
        for i in range(n):
            if i!=k:
                factor=rows[i][k];rows[i]=[x-factor*y for x,y in zip(rows[i],rows[k])]
    return [row[-1] for row in rows]


def verify(payload):
    c=payload['contract'];d=c['d']
    require(isinstance(d,int) and d>=2 and c['m']==2 and c['s']==1,'unsupported dimensions or sparsity')
    names=['A','a_min','T','R','J','H1','H2','K','h','C0','eta']
    q={name:F(c[name]) for name in names}
    A,amin,T,R,J,H1,H2,K,h,C0,eta=[q[name] for name in names]
    require(A>=amin>0 and T>=0 and R>=0 and J>=0 and eta>=0,'invalid class bounds')
    require(H2>H1>T and h>0,'invalid excitation geometry')
    require(K>max(H2,T+A*(H2+T)) and C0>T+h,'suppression or calibration condition failed')
    tr=payload['transcript'];cal=matrix(tr['calibration'],d+1,2)
    require(isinstance(tr['source'],list) and len(tr['source'])==d,'source count differs')
    source=[matrix(v,2,2) for v in tr['source']]
    chat=[[(cal[j+1][r]-cal[0][r])/h for j in range(d)] for r in range(2)]
    bhat=subtract(cal[0],mv(chat,[-C0]*d))
    em=2*d*eta/h;eb=eta+C0*em;dh=H2-H1
    eu=2*eta/dh+em*max(F(1),A)
    ev=2*H1*eta/dh+eta+eb+em*(K+A*T)
    margins=[]
    for i,j in combinations(range(d),2):
        a,b=chat[0][i],chat[0][j];c0,d0=chat[1][i],chat[1][j]
        det=a*d0-b*c0
        require(det!=0,'a two-column readout minor is singular')
        inverse=[[d0/det,-b/det],[-c0/det,a/det]]
        margins.append(1/mnorm(inverse))
    kappa=min(margins);ea=2*eu/kappa;et=(2*ev/kappa+T*ea)/amin
    require(ea<amin,'support margin fails')
    candidate=payload['candidate'];gain=matrix(candidate['gains'],d,d)
    product=matrix(candidate['v_products'],d,d);weight=vec(candidate['weight'],d)
    threshold=[[F(0)]*d for _ in range(d)]
    for k in range(d):
        acol=[gain[i][k] for i in range(d)];vcol=[product[i][k] for i in range(d)]
        require(acol[k]==0 and vcol[k]==0,'nonzero diagonal')
        require(sum(a!=0 for a in acol)<=1,'out-degree exceeds declared sparsity')
        for i,(a,v) in enumerate(zip(acol,vcol)):
            require(a==0 or amin<=abs(a)<=A,'gain outside permitted signed intervals')
            require(a!=0 or v==0,'threshold product outside gain support')
            require(abs(v)<=A*T,'threshold-product bound fails')
            if a:threshold[i][k]=max(-T,min(T,v/a))
        y1,y2=source[k];slope=[(v-u)/dh for u,v in zip(y1,y2)]
        u=subtract(slope,[row[k] for row in chat])
        v=[H1*slope[r]-y1[r]-K*sum(chat[r])+K*chat[r][k]+bhat[r] for r in range(2)]
        require(norm(subtract(mv(chat,acol),u))<=eu,'gain measurement residual fails')
        require(norm(subtract(mv(chat,vcol),v))<=ev,'threshold-product residual fails')
    U=[[min(A,abs(gain[i][k])+ea) if gain[i][k] else F(0) for k in range(d)] for i in range(d)]
    require(all(x>0 for x in weight),'weight not strictly positive')
    require(all(x<y for x,y in zip(mv(U,weight),weight)),'weighted contraction fails')
    resolvent=[[F(i==j)-U[i][j] for j in range(d)] for i in range(d)]
    K0=max(R,J);B=solve(resolvent,[K0+T*sum(row) for row in U])
    require(all(x>=K0 for x in B),'state supersolution is invalid')
    forcing=[sum(ea*(B[k]+T)+abs(gain[i][k])*et for k in range(d) if gain[i][k]) for i in range(d)]
    D=solve(resolvent,forcing)
    require(all(x>=0 for x in D),'negative state-error bound')
    radius=em*norm(B)+mnorm(chat)*norm(D)+eb
    return {'status':'ACCEPT','arithmetic':'EXACT_RATIONAL','latent_dimension':d,'output_dimension':2,'sparsity':1,'queries':3*d+1,'readout':[[str(x) for x in row] for row in chat],'offset':[str(x) for x in bhat],'thresholds':[[str(x) for x in row] for row in threshold],'restricted_margin_lower':str(kappa),'gain_error_upper':str(ea),'threshold_error_upper':str(et),'state_bound':[str(x) for x in B],'state_error_bound':[str(x) for x in D],'uniform_output_error_upper':str(radius),'producer_imported':False,'ground_truth_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path);args=parser.parse_args()
    try:
        result=verify(json.loads(args.certificate.read_text()))
    except (ValueError,KeyError,TypeError,ZeroDivisionError,OSError) as error:
        print(json.dumps({'status':'REJECT','reason':str(error)}));return 1
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
