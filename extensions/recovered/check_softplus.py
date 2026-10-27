from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq

rng=np.random.default_rng(20261026)
d=4;A=.3;amin=.1;T=1.;H1=2.;H2=4.;rho0=.35;rho=.5;Mmax=2.
P=lambda t:np.logaddexp(0,t)
sigmoid=lambda t:1/(1+np.exp(-t))
def ratio(t):return float(P(H2-t)/P(H1-t))
qgrid=np.exp(np.linspace(H1-T,H2+T,201))
lower_hprime=min(qlo*(qlo-np.log1p(qlo))/((1+qhi)**2*np.log1p(qhi)**2) for qlo,qhi in zip(qgrid,qgrid[1:]))
mr=(H2-H1)*lower_hprime
mf=amin*P(H1-T)
a=np.zeros((d,d));tau=np.zeros((d,d))
for k in range(d):
    a[(k+1)%d,k]=rng.choice([-1,1])*rng.uniform(.15,.25)
    tau[(k+1)%d,k]=rng.uniform(-T,T)
M=np.eye(d)+.1*np.roll(np.eye(d),1,axis=1);b=rng.normal(0,.1,d)
assert np.linalg.norm(M,np.inf)<=Mmax and np.linalg.norm(a,np.inf)<=rho0
def equilibrium(x,aa,tt,clamps=None,lam=None,replacements=None):
    z=np.zeros(d)
    for _ in range(300):
        gates=P(z[None,:]-tt)
        if replacements is not None:gates=np.where(np.isfinite(replacements),replacements,gates)
        nxt=x+np.sum(aa*gates*(1 if lam is None else lam),axis=1)
        if clamps is not None:nxt=np.where(np.isfinite(clamps),clamps,nxt)
        if np.max(np.abs(nxt-z))<1e-13:return nxt
        z=nxt
    raise AssertionError('no convergence')
records=[]
for S in [8.,12.,16.,20.]:
    h=S;C0=2*S+1;K=S+A*P(H2+T)+1
    tail=A*np.exp(-S+T)
    effective=Mmax*(d-1)*tail/(1-rho0)
    probes=[np.full(d,-C0)]
    for j in range(d):
        x=probes[0].copy();x[j]+=h;probes.append(x)
    for k in range(d):
        for H in [H1,H2]:
            x=np.full(d,-K);x[k]=H;probes.append(x)
    y=[M@equilibrium(x,a,tau)+b for x in probes]
    Mh=np.column_stack([(y[j+1]-y[0])/h for j in range(d)]);bh=y[0]-Mh@probes[0]
    em=2*d*effective/h;eb=effective+C0*em;inv=np.linalg.inv(Mh)
    Cstar=max(H2,K+A*P(H2+T));delta=np.linalg.norm(inv,np.inf)*(effective+eb+Cstar*em)
    record={'suppression':S,'vector_queries':len(probes),'effective_output_error_upper':float(effective),'source_value_error_upper':float(delta)}
    if delta>=mf/2:
        record['status']='UNKNOWN_MARGIN';records.append(record);continue
    et=delta*(1+ratio(T))/((mf-delta)*mr)
    ea=(delta+A*et)/P(H1-T)
    ah=np.zeros_like(a);th=np.zeros_like(tau)
    for k in range(d):
        z1=inv@(y[d+1+2*k]-bh);z2=inv@(y[d+2+2*k]-bh)
        v1=z1+K;v2=z2+K
        for i in range(d):
            if i==k or abs(v1[i])<=mf/2:continue
            rr=np.clip(v2[i]/v1[i],ratio(-T),ratio(T))
            if rr==ratio(-T):threshold=-T
            elif rr==ratio(T):threshold=T
            else:threshold=brentq(lambda t:ratio(t)-rr,-T,T,xtol=1e-14)
            gain=v1[i]/P(H1-threshold)
            ah[i,k]=np.sign(gain)*np.clip(abs(gain),amin,A);th[i,k]=threshold
    assert np.array_equal(np.sign(ah),np.sign(a))
    gain_error=np.max(np.abs(ah-a));threshold_error=np.max(np.abs((th-tau)[a!=0]))
    assert gain_error<=ea+1e-11 and threshold_error<=et+1e-11
    assert np.linalg.norm(ah,np.inf)<=rho
    B=max(1.,(1+rho*P(T))/(1-rho));D=((d-1)*ea*(B+P(T))+rho*et)/(1-rho)
    bound=em*B+np.linalg.norm(Mh,np.inf)*D+eb;observed=0.
    for _ in range(50):
        x=rng.uniform(-1,1,d);clamps=np.where(rng.random(d)<.2,rng.uniform(-1,1,d),np.nan)
        lam=rng.uniform(0,1,(d,d));repl=np.where(rng.random((d,d))<.2,rng.uniform(0,1,(d,d)),np.nan)
        true=M@equilibrium(x,a,tau,clamps,lam,repl)+b
        estimated=Mh@equilibrium(x,ah,th,clamps,lam,repl)+bh
        observed=max(observed,float(np.max(np.abs(true-estimated))))
    assert observed<=bound+1e-11
    record.update(status='PASS_NUMERICAL',gain_error_upper=float(ea),maximum_gain_error=float(gain_error),threshold_error_upper=float(et),maximum_threshold_error=float(threshold_error),uniform_intervention_error_upper=float(bound),maximum_sampled_intervention_error=observed,intervention_tests=50)
    records.append(record)
result={'status':'PASS_NUMERICAL','interpretation':'Fixed 3d+1 probes give certified approximate softplus reconstruction under tail, conditioning and contraction bounds; this check uses floating arithmetic, not an interval proof implementation.','ratio_derivative_positive_lower':float(mr),'records':records}
(Path(__file__).resolve().parent/'generated/softplus_extension_check.json').write_text(json.dumps(result,indent=2))
p=Path(__file__).resolve().parent/'generated/frontier_verification.json';all_results=json.loads(p.read_text());all_results['softplus_extension']=result;p.write_text(json.dumps(all_results,indent=2))
print(json.dumps(result,indent=2))
