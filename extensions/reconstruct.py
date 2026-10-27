"""Fresh deterministic rational fixtures; historical summaries are not regenerated data."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import random,json
from check_sparse import check


def solve(a,b):
    # Producer's independent Gaussian elimination, separate from checker inverse.
    n=len(b); a=[list(r)+[v] for r,v in zip(a,b)]
    for j in range(n):
        k=next(k for k in range(j,n) if a[k][j]); a[j],a[k]=a[k],a[j]
        q=a[j][j]; a[j]=[v/q for v in a[j]]
        for i in range(j+1,n):
            q=a[i][j]; a[i]=[v-q*w for v,w in zip(a[i],a[j])]
    x=[F(0)]*n
    for i in reversed(range(n)): x[i]=a[i][-1]-sum(a[i][j]*x[j] for j in range(i+1,n))
    return x


def state(gain,tau,x,clamps=None,atten=None,gates=None):
    d=len(x); clamps=clamps or {}; atten=atten or {}; gates=gates or {}
    edges=[(i,k) for i in range(d) for k in range(d) if gain[i][k] and i not in clamps and (i,k) not in gates]
    solutions=set()
    for bits in product([0,1],repeat=len(edges)):
        a=[[F(i==k) for k in range(d)] for i in range(d)]; b=list(x)
        for i in range(d):
            if i in clamps: b[i]=clamps[i]; continue
            for k in range(d):
                if (i,k) in gates: b[i]+=gain[i][k]*atten.get((i,k),1)*gates[i,k]
        for (i,k),on in zip(edges,bits):
            q=gain[i][k]*atten.get((i,k),1)*on
            a[i][k]-=q; b[i]-=q*tau[i][k]
        try: z=solve(a,b)
        except (StopIteration,ZeroDivisionError): continue
        if all((z[k]>=tau[i][k] if on else z[k]<=tau[i][k]) for (i,k),on in zip(edges,bits)):
            solutions.add(tuple(z))
    if len(solutions)!=1: raise ValueError('state not uniquely solved')
    return list(solutions.pop())


def output(C,b,z): return [sum(v*w for v,w in zip(row,z))+off for row,off in zip(C,b)]


def fit_scalar(column,target,tol,lo,hi):
    for c,v in zip(column,target):
        if not c:
            if abs(v)>tol: return None
        else:
            ends=sorted([(v-tol)/c,(v+tol)/c]); lo=max(lo,ends[0]); hi=min(hi,ends[1])
    return (lo+hi)/2 if lo<=hi else None


def run(directory):
    directory.mkdir(parents=True,exist_ok=True); rng=random.Random(20261026)
    reports=[]
    for trial in range(12):
        d=4; A=F(2,5); amin=F(1,10); T=F(1); eta=F(0) if trial==0 else F(1,10**8)
        C=[[F(1)]*d,list(map(F,range(d)))]; b=[F(2,7),F(-1,5)]
        gain=[[F(0)]*d for _ in range(d)]; tau=[[F(0)]*d for _ in range(d)]
        for k in range(d):
            i=(k+1)%d; gain[i][k]=F(rng.choice([-3,-2,2,3]),10); tau[i][k]=F(rng.choice([-2,-1,0,1,2]),4)
        C0=F(5,2); h=F(1); K=F(5); H1=F(2); H2=F(3)
        probes=[[-C0]*d]
        for j in range(d):
            x=[-C0]*d; x[j]+=h; probes.append(x)
        for k in range(d):
            for H in [H1,H2]:
                x=[-K]*d; x[k]=H; probes.append(x)
        obs=[[y+eta*F(rng.randint(-10,10),10) for y in output(C,b,state(gain,tau,x))] for x in probes]
        Chat=[[(obs[j+1][i]-obs[0][i])/h for j in range(d)] for i in range(2)]
        bhat=[obs[0][i]+C0*sum(Chat[i]) for i in range(2)]
        em=2*d*eta/h; eb=eta+C0*em; eu=2*eta/(H2-H1)+em*max(1,A)
        ev=2*H1*eta/(H2-H1)+eta+eb+em*(K+A*T)
        ahat=[[F(0)]*d for _ in range(d)]; vhat=[[F(0)]*d for _ in range(d)]
        for k in range(d):
            y1,y2=obs[d+1+2*k:d+3+2*k]; ell=[(y2[i]-y1[i])/(H2-H1) for i in range(2)]
            u=[ell[i]-Chat[i][k] for i in range(2)]
            w=[H1*ell[i]-y1[i]-K*sum(Chat[i])+K*Chat[i][k]+bhat[i] for i in range(2)]
            fits=[]
            for i in range(d):
                if i==k: continue
                column=[r[i] for r in Chat]
                for lo,hi in [(-A,-amin),(amin,A)]:
                    av=fit_scalar(column,u,eu,lo,hi)
                    if av is not None:
                        vv=fit_scalar(column,w,ev,-A*T,A*T)
                        if vv is not None: fits.append((i,av,vv))
            if len(fits)!=1: raise ValueError('ambiguous fixture fit')
            i,av,vv=fits[0]; ahat[i][k]=av; vhat[i][k]=vv
        record=dict(schema='tcabb.sparse.v1',d=d,m=2,s=1,
            bounds=dict(A=A,amin=amin,T=T,R=F(1),J=F(1),H1=H1,H2=H2,K=K,h=h,C0=C0,eta=eta),
            observations=obs,gain=ahat,product=vhat,weight=[F(1)]*d)
        path=directory/f'case_{trial:02d}.json'; path.write_text(json.dumps(record,default=str,indent=2)+'\n')
        checked=check(json.loads(path.read_text())); th=checked['tau']; bound=checked['radius']
        assert all(abs(ahat[i][k]-gain[i][k])<=checked['gain_error'] for i in range(d) for k in range(d))
        assert all(abs(th[i][k]-tau[i][k])<=checked['threshold_error'] for i in range(d) for k in range(d) if gain[i][k])
        maximum=F(0)
        for _ in range(20):
            x=[F(rng.randint(-4,4),4) for _ in range(d)]
            clamps={i:F(rng.randint(-4,4),4) for i in range(d) if rng.random()<.2}
            atten={(i,k):rng.choice([F(0),F(1,2),F(1)]) for i in range(d) for k in range(d) if gain[i][k]}
            gates={(i,k):F(rng.randint(0,4),4) for i,k in atten if rng.random()<.15}
            y=output(C,b,state(gain,tau,x,clamps,atten,gates))
            yh=output(checked['C'],checked['b'],state(ahat,th,x,clamps,atten,gates))
            error=max(abs(v-w) for v,w in zip(y,yh)); assert error<=bound; maximum=max(maximum,error)
        reports.append(dict(case=path.name,radius=bound,maximum_actual_error=maximum,comparisons=20))
    return dict(status='PASS',provenance='Fresh audit fixtures, not original missing producer replay',cases=reports,comparisons=240)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=run(root/'extensions/certificates')
    (root/'reports/sparse_reconstruction.json').write_text(json.dumps(result,default=str,indent=2)+'\n')
    print('PASS: 12 fresh rational fixtures, 240 intervention comparisons')
