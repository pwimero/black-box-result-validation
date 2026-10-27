"""Independent checks of proposed black-box recovery extensions.

Exact rational checks are distinguished from numerical sanity checks.
These checks supplement the accompanying proofs; they do not replace them.
"""
from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog


def solve(matrix, rhs):
    n = len(matrix)
    a = [[F(v) for v in row]+[F(rhs[i])] for i,row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None: raise ValueError('singular')
        a[j],a[pivot]=a[pivot],a[j]
        scale=a[j][j];a[j]=[v/scale for v in a[j]]
        for i in range(n):
            if i!=j:
                factor=a[i][j]
                a[i]=[u-factor*v for u,v in zip(a[i],a[j])]
    return [row[-1] for row in a]


def mv(m,v):return [sum(x*y for x,y in zip(row,v)) for row in m]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def add(a,b):return [x+y for x,y in zip(a,b)]
def ninf(v):return max(map(abs,v),default=F(0))
def matnorm(m):return max(sum(map(abs,row)) for row in m)


def scalar_chebyshev(column, target, lo, hi):
    lines=[(sign*c,-sign*y) for c,y in zip(column,target) for sign in (-1,1)]
    candidates={F(lo),F(hi)}
    for (a,b),(c,d) in combinations(lines,2):
        if a!=c:
            v=(d-b)/(a-c)
            if lo<=v<=hi:candidates.add(v)
    return min(((ninf(sub([c*v for c in column],target)),v) for v in candidates),key=lambda x:x[0])


def sparse_one(c,target,excluded,A,amin):
    best=(ninf(target),[F(0)]*len(c[0]))
    for i in range(len(c[0])):
        if i==excluded:continue
        column=[row[i] for row in c]
        for lo,hi in ((-A,-amin),(amin,A)):
            residual,value=scalar_chebyshev(column,target,lo,hi)
            if residual<best[0]:
                vector=[F(0)]*len(c[0]);vector[i]=value;best=(residual,vector)
    return best


def restricted_margin_2(c):
    bounds=[]
    for i,j in combinations(range(len(c[0])),2):
        a,b=c[0][i],c[0][j];q,r=c[1][i],c[1][j]
        det=a*r-b*q
        if not det:return F(0)
        inv=[[r/det,-b/det],[-q/det,a/det]]
        bounds.append(1/matnorm(inv))
    return min(bounds)


def relu_state(x,a,tau,clamp=None,atten=None,replacement=None):
    d=len(x);clamp={} if clamp is None else clamp
    atten={} if atten is None else atten
    replacement={} if replacement is None else replacement
    active_edges=[(i,k) for i in range(d) for k in range(d) if a[i][k] and i not in clamp and (i,k) not in replacement]
    for bits in product((0,1),repeat=len(active_edges)):
        active=dict(zip(active_edges,bits))
        m=[[F(i==j) for j in range(d)] for i in range(d)];rhs=list(x)
        for i in range(d):
            if i in clamp:rhs[i]=F(clamp[i]);continue
            for k in range(d):
                coefficient=a[i][k]*atten.get((i,k),F(1))
                if not coefficient:continue
                if (i,k) in replacement:rhs[i]+=coefficient*replacement[i,k]
                elif active.get((i,k),0):
                    m[i][k]-=coefficient;rhs[i]-=coefficient*tau[i][k]
        z=solve(m,rhs)
        if all((z[k]>=tau[i][k] if bit else z[k]<=tau[i][k]) for (i,k),bit in active.items()):return z
    raise AssertionError('No active set')


ARTIFACT_ROOT=Path(__file__).resolve().parent/'generated'
ARTIFACT_ROOT.mkdir(exist_ok=True)
CERT_DIR=ARTIFACT_ROOT/'sparse_certificates'
CERT_DIR.mkdir(exist_ok=True)
rng=np.random.default_rng(20261026)
d=4;C=[[F(1)]*d,[F(i) for i in range(d)]];b=[F(2,7),F(-1,5)]
A=F(2,5);amin=F(1,10);T=F(1);H1=F(2);H2=F(3);K=F(5);h=F(1);C0=F(5,2)
results={}
noise_records=[];total_interventions=0
for trial in range(12):
    a=[[F(0)]*d for _ in range(d)];tau=[[F(0)]*d for _ in range(d)]
    for k in range(d):
        i=(k+1)%d;a[i][k]=F(int(rng.choice([-3,-2,2,3])),10);tau[i][k]=F(int(rng.integers(-2,3)),4)
    probes=[[-C0]*d]
    for j in range(d):
        p=[-C0]*d;p[j]+=h;probes.append(p)
    for k in range(d):
        for H in (H1,H2):
            p=[-K]*d;p[k]=H;probes.append(p)
    eta=F(0) if trial==0 else F(1,10**8)
    ys=[]
    for p in probes:
        z=relu_state(p,a,tau);y=add(mv(C,z),b)
        ys.append([v+eta*F(int(rng.integers(-10,11)),10) for v in y])
    chat=[[ (ys[j+1][r]-ys[0][r])/h for j in range(d)] for r in range(2)]
    bhat=sub(ys[0],mv(chat,probes[0]));em=2*d*eta/h;eb=eta+C0*em
    eu=2*eta/(H2-H1)+em*max(F(1),A)
    ev=2*H1*eta/(H2-H1)+eta+eb+em*(K+A*T)
    kap=restricted_margin_2(chat);ea=2*eu/kap;et=(2*ev/kap+T*ea)/amin
    assert kap>0 and ea<amin
    ahat=[[F(0)]*d for _ in range(d)];that=[[F(0)]*d for _ in range(d)];vproducts=[[F(0)]*d for _ in range(d)]
    for k in range(d):
        y1,y2=ys[d+1+2*k:d+3+2*k]
        slope=[(q-p)/(H2-H1) for p,q in zip(y1,y2)]
        u=sub(slope,[row[k] for row in chat])
        v=[H1*slope[r]-y1[r]-K*sum(chat[r])+K*chat[r][k]+bhat[r] for r in range(2)]
        residual,avec=sparse_one(chat,u,k,A,amin)
        assert residual<=eu
        for i in range(d):
            ahat[i][k]=avec[i]
            assert abs(avec[i]-a[i][k])<=ea
            assert (avec[i]>0)-(avec[i]<0)==(a[i][k]>0)-(a[i][k]<0)
            if avec[i]:
                residual,w=scalar_chebyshev([row[i] for row in chat],v,-A*T,A*T)
                assert residual<=ev
                vproducts[i][k]=w
                that[i][k]=min(T,max(-T,w/avec[i]))
                assert abs(that[i][k]-tau[i][k])<=et
    rho=F(1,2);assert matnorm(ahat)<=rho
    B=max(F(1),(F(1)+rho*T)/(1-rho))
    D=((d-1)*ea*(B+T)+rho*et)/(1-rho)
    certificate=em*B+matnorm(chat)*D+eb
    U=[[min(A,abs(ahat[i][k])+ea) if ahat[i][k] else F(0) for k in range(d)] for i in range(d)]
    weight=solve([[F(i==j)-U[i][j] for j in range(d)] for i in range(d)],[F(1)]*d)
    encode=lambda rows:[[str(v) for v in row] for row in rows]
    artifact={'contract':{'d':d,'m':2,'s':1,**{name:str(value) for name,value in [('A',A),('a_min',amin),('T',T),('R',F(1)),('J',F(1)),('H1',H1),('H2',H2),('K',K),('h',h),('C0',C0),('eta',eta)]}},'transcript':{'calibration':encode(ys[:d+1]),'source':[encode(ys[d+1+2*k:d+3+2*k]) for k in range(d)]},'candidate':{'gains':encode(ahat),'v_products':encode(vproducts),'weight':[str(v) for v in weight]}}
    (CERT_DIR/f'case_{trial:02d}.json').write_text(json.dumps(artifact,indent=2))
    ratios=[];maximum_actual_error=F(0)
    for q in range(20):
        x=[F(int(rng.integers(-4,5)),4) for _ in range(d)]
        clamps={i:F(int(rng.integers(-4,5)),4) for i in range(d) if rng.random()<.2}
        atten={(i,k):F(int(rng.integers(0,3)),2) for i in range(d) for k in range(d)}
        replacements={(i,k):F(int(rng.integers(0,5)),4) for i in range(d) for k in range(d) if rng.random()<.15}
        z=relu_state(x,a,tau,clamps,atten,replacements);zh=relu_state(x,ahat,that,clamps,atten,replacements)
        err=ninf(sub(add(mv(C,z),b),add(mv(chat,zh),bhat)))
        assert err<=certificate
        maximum_actual_error=max(maximum_actual_error,err)
        total_interventions+=1
        if certificate:ratios.append(float(err/certificate))
    noise_records.append({'trial':trial,'noise':str(eta),'restricted_margin_lower':str(kap),'gain_error_upper':str(ea),'threshold_error_upper':str(et),'maximum_observed_error_over_bound':max(ratios,default=0),'maximum_actual_error':str(maximum_actual_error)})
results['sparse_readout_extension']={'status':'PASS_EXACT_RATIONAL','latent_dimension':d,'output_dimension':2,'maximum_out_degree':1,'vector_queries':3*d+1,'models':12,'intervention_comparisons':total_interventions,'noiseless_gain_and_threshold_recovery_exact':True,'noise_records':noise_records}

# Exact known-basis extension: three hinges per edge, two observed outputs.
knots=[F(0),F(1),F(2)];levels=[F(1,2),F(3,2),F(3)]
Phi=[[max(F(0),t-k) for k in knots] for t in levels]
coeff=[ [F(int(rng.choice([-1,1])),20) for _ in knots] for _ in range(d)]
basis_calibration=[add(mv(C,[-C0]*d),b)]
for j in range(d):
    state=[-C0]*d;state[j]+=h
    basis_calibration.append(add(mv(C,state),b))
C_basis=[[(basis_calibration[j+1][r]-basis_calibration[0][r])/h for j in range(d)] for r in range(2)]
b_basis=sub(basis_calibration[0],mv(C_basis,[-C0]*d))
assert C_basis==C and b_basis==b
recovered=[]
for k in range(d):
    values=[]
    for t in levels:
        value=sum(c*max(F(0),t-q) for c,q in zip(coeff[k],knots))
        z=[-K]*d;z[k]=t;z[(k+1)%d]+=value
        y=add(mv(C,z),b)
        compressed=[y[r]+K*sum(C_basis[r])-(t+K)*C_basis[r][k]-b_basis[r] for r in range(2)]
        residual,vector=sparse_one(C_basis,compressed,k,F(2),F(0))
        assert residual==0 and vector[(k+1)%d]==value
        values.append(vector[(k+1)%d])
    beta=solve(Phi,values);assert beta==coeff[k];recovered.append([str(v) for v in beta])
results['known_hinge_basis_extension']={'status':'PASS_EXACT_RATIONAL','basis_size':3,'latent_dimension':d,'output_dimension':2,'vector_queries':d+1+d*len(levels),'recovered_coefficients':recovered,'unknown_readout_recovered_from_calibration':True}

# Numerical LP check of a single joint decoder into l_infinity, including
# Holder moduli and closest points of bounded-noise observation intervals.
joint=[]
r=np.array([-1.,.2,.7,2.]);semantic=np.array([[0.,2.,-1.],[1.,0.,2.],[-1.,1.,.5],[2.,-1.,1.]])
for beta in [1.,.5,1/3]:
    for eta in [0.,.1,.4]:
        observations=[]
        for value in r:observations +=[value-eta,value+eta]
        for left,right in combinations(r,2):
            lo=max(left-eta,right-eta);hi=min(left+eta,right+eta)
            if lo<=hi:observations.append(lo)
        obs=np.unique(observations);m=len(obs);k=semantic.shape[1];nvar=m*k+1;rows=[];rhs=[]
        for i,value in enumerate(r):
            for j,y in enumerate(obs):
                if abs(value-y)<=eta+1e-12:
                    for c in range(k):
                        for sign in [-1,1]:
                            row=np.zeros(nvar);row[j*k+c]=sign;row[-1]=-1;rows.append(row);rhs.append(sign*semantic[i,c])
        for i,j in combinations(range(m),2):
            for c in range(k):
                for sign in [-1,1]:
                    row=np.zeros(nvar);row[i*k+c]=sign;row[j*k+c]=-sign;rows.append(row);rhs.append(2*abs(obs[i]-obs[j])**beta)
        objective=np.zeros(nvar);objective[-1]=1
        sol=linprog(objective,A_ub=rows,b_ub=rhs,bounds=[(None,None)]*(nvar-1)+[(0,None)],method='highs')
        assert sol.success
        defect=max(max(0,np.max(np.abs(semantic[i]-semantic[j]))-2*max(abs(r[i]-r[j])-2*eta,0)**beta) for i in range(len(r)) for j in range(len(r)))/2
        assert abs(sol.fun-defect)<1e-7
        joint.append({'holder_exponent':beta,'noise':eta,'linear_program_optimum':sol.fun,'pair_defect_formula':float(defect)})
results['joint_modulus_identity']={'status':'PASS_NUMERICAL_LP','cases':joint}

# Contractive nonseparable field, globally rho-Lipschitz, with a known zero.
rho=.4;B=1.;R=.6;delta=.001;mesh=np.linspace(-1.4,1.4,17);fill=(mesh[1]-mesh[0])/2
def field(z):return np.array([.2*np.tanh(z[0]+z[1]),.15*np.sin(z[0]-z[1])])
def fixed_point(x,func,clamp=None,refs=None):
    z=np.zeros(2)
    for _ in range(200):
        if refs is None:val=func(z)
        else:
            val=np.array([func(np.array([refs[i,j] if np.isfinite(refs[i,j]) else z[j] for j in range(2)]))[i] for i in range(2)])
        nxt=x+val
        if clamp is not None:nxt=np.where(np.isfinite(clamp),clamp,nxt)
        if np.max(np.abs(nxt-z))<1e-13:return nxt
        z=nxt
    raise AssertionError('fixed point failed')
X=np.array(list(product(mesh,repeat=2)));Z=np.array([fixed_point(x,field) for x in X]);Zhat=Z+rng.uniform(-delta,delta,Z.shape);V=Zhat-X
eps0=(1+rho)*delta
def approx_field(z):
    distances=np.max(np.abs(Zhat-z),axis=1)
    lo=np.maximum(np.max(V-eps0-rho*distances[:,None],axis=0),-rho*np.max(np.abs(z)))
    hi=np.minimum(np.min(V+eps0+rho*distances[:,None],axis=0),rho*np.max(np.abs(z)))
    assert np.all(lo<=hi+1e-11)
    return (lo+hi)/2
eps=eps0+rho*(fill/(1-rho)+delta);bound=eps/(1-rho)
largest_field_error=0.;largest_state_error=0.
for _ in range(150):
    z=rng.uniform(-B,B,2);largest_field_error=max(largest_field_error,float(np.max(np.abs(field(z)-approx_field(z)))))
    x=rng.uniform(-R,R,2);clamp=np.where(rng.random(2)<.2,rng.uniform(-B,B,2),np.nan)
    refs=np.where(rng.random((2,2))<.3,rng.uniform(-B,B,(2,2)),np.nan)
    true=fixed_point(x,field,clamp,refs);estimated=fixed_point(x,approx_field,clamp,refs)
    largest_state_error=max(largest_state_error,float(np.max(np.abs(true-estimated))))
assert largest_field_error<=eps+1e-10 and largest_state_error<=bound+1e-10
assert np.max(np.abs(approx_field(np.zeros(2))))<1e-12
results['general_contractive_field']={'status':'PASS_NUMERICAL','dimension':2,'oracle_queries':len(X),'rho':rho,'state_observation_error':delta,'input_grid_fill_radius':float(fill),'field_error_upper':float(eps),'maximum_sampled_field_error':largest_field_error,'uniform_intervention_error_upper':float(bound),'maximum_sampled_intervention_error':largest_state_error,'intervention_tests':150,'known_zero_preserved':True}

# Explicit indistinguishable scalar-readout stars.
results['partial_readout_counterexample']={'status':'CHECKED_ALGEBRAICALLY','models':'z1=x1,z2=x2+a[x1]+,z3=x3-a[x1]+ versus no edges; y=z2+z3','ordinary_outputs':'x2+x3 in both models for every input','edge_ablation':'Ablating 1->2 reveals -a[x1]+ in the first model.'}

(ARTIFACT_ROOT/'frontier_verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps({k:({x:y for x,y in v.items() if x not in ['noise_records','cases','recovered_coefficients']}) for k,v in results.items()},indent=2))
