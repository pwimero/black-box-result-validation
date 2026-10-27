from pathlib import Path
from fractions import Fraction as F
import copy,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extensions'))
from check_sparse import check,inverse,mv,norm
from reconstruct import state,output,solve
from check_known_anchor import check as check_anchor


@pytest.mark.parametrize('case',range(12))
def test_fresh_certificates(case):
    cert=json.loads((ROOT/f'extensions/certificates/case_{case:02d}.json').read_text())
    result=check(cert)
    assert result['queries']==13 and result['radius']>=0
    if case==0: assert result['radius']==0


@pytest.mark.parametrize('mutation',['diagonal','noise','weight','geometry','product'])
def test_adversarial_mutations(mutation):
    cert=json.loads((ROOT/'extensions/certificates/case_01.json').read_text())
    if mutation=='diagonal': cert['gain'][0][0]='1/5'
    if mutation=='noise': cert['bounds']['eta']='0'
    if mutation=='weight': cert['weight'][0]='0'
    if mutation=='geometry': cert['bounds']['K']='0'
    if mutation=='product': cert['product'][0][1]='1'
    with pytest.raises(ValueError): check(cert)


def test_singular_readout_rejected():
    cert=json.loads((ROOT/'extensions/certificates/case_00.json').read_text())
    # Replace all second output coordinates by first: destroys restricted injectivity.
    for row in cert['observations']: row[1]=row[0]
    with pytest.raises(ValueError,match='singular'): check(cert)


def test_weighted_resolvent_attained():
    U=[[F(0),F(3)],[F(1,10),F(0)]]
    assert mv(U,[F(5),F(1)])==[F(3),F(1,2)]
    inv=inverse([[F(i==j)-U[i][j] for j in range(2)] for i in range(2)])
    assert mv(inv,[F(1,100),F(1,50)])==[F(1,10),F(3,100)]


def test_negative_feedback_uncontracted_interventions():
    gain=[[F(0),F(2)],[F(-2),F(0)]]; tau=[[F(0),F(1,4)],[F(-1,2),F(0)]]
    grid=[F(i,4) for i in range(-4,5)]
    for x in [(a,b) for a in grid for b in grid]:
        z=state(gain,tau,x)
        assert z[0]==x[0]+2*max(F(0),z[1]-F(1,4))
        assert z[1]==x[1]-2*max(F(0),z[0]+F(1,2))
        other=state(gain,tau,[x[0]+F(1,7),x[1]-F(1,11)])
        assert norm([a-b for a,b in zip(z,other)])<=3*F(1,7)
        for clamps,atten,gates in [({0:F(1)}, {},{}), ({},{(0,1):F(1,2)},{(1,0):F(3,4)}),
                                   ({1:F(-1)}, {(0,1):F(0)}, {})]:
            state(gain,tau,x,clamps,atten,gates)


def test_basis_identification():
    knots=[F(0),F(1),F(2)]; levels=[F(1,2),F(3,2),F(3)]
    Phi=[[max(F(0),t-k) for k in knots] for t in levels]
    beta=[F(-1,20),F(1,20),F(1,20)]
    assert solve(Phi,mv(Phi,beta))==beta


@pytest.mark.parametrize('beta',[1,.5,1/3])
@pytest.mark.parametrize('eta',[0,.1,.4,1])
def test_joint_minimax_against_independent_linear_program(beta,eta):
    import numpy as np
    from scipy.optimize import linprog
    response=np.array([-1.,.4,2.]); target=np.array([[0.,2.],[2.,-.5],[1.,3.]])
    sites=sorted(set(float(r+e) for r in response for e in [-eta,0,eta]))
    # One common two-coordinate decoder with prescribed modulus at all noise sites.
    constraints=[]; rhs=[]; n=2*len(sites)+1
    for j,y in enumerate(sites):
        for w,r in enumerate(response):
            if abs(y-r)<=eta+1e-12:
                for c in range(2):
                    for sign in [-1,1]:
                        row=np.zeros(n); row[2*j+c]=sign; row[-1]=-1
                        constraints.append(row); rhs.append(sign*target[w,c])
        for k,z in enumerate(sites):
            for c in range(2):
                row=np.zeros(n); row[2*j+c]=1; row[2*k+c]-=1
                constraints.append(row); rhs.append(abs(y-z)**beta)
    objective=np.zeros(n); objective[-1]=1
    result=linprog(objective,A_ub=constraints,b_ub=rhs,bounds=[(None,None)]*(n-1)+[(0,None)],method='highs')
    defect=max(0.,max(np.max(abs(target[i]-target[j]))-max(0,abs(response[i]-response[j])-2*eta)**beta
                      for i in range(3) for j in range(3)))/2
    assert result.success and result.fun==pytest.approx(defect,abs=1e-8)


def test_known_anchor_gradients():
    path=ROOT/'legacy/results/continuous_triangle_duals.json.gz'
    if not path.exists(): path=ROOT/'results/continuous_triangle_duals.json.gz'
    result=check_anchor(path)
    assert sum(a['cells'] for a in result['anchors'])==1689
    assert all(a['ratio']<=6 for a in result['anchors'])


def test_whole_edge_replacements_need_additional_state_bound():
    # Why the noisy radius must explicitly restrict its intervention family.
    U=[[F(0),F(1,10)],[F(0),F(0)]]
    B=mv(inverse([[F(i==j)-U[i][j] for j in range(2)] for i in range(2)]),[F(11,10),F(1)])
    replaced_output_state=F(100)  # x=0, whole edge output replaced by 100.
    assert replaced_output_state>B[0]


def test_integer_normalization_matches_reference_fraction_arithmetic():
    import random
    from check_finite_tables import stable_error,load_checker
    path=ROOT/'legacy/reference_checker/check_jmlr_prospective_suite.py'
    if not path.exists(): path=ROOT/'reference_checker/check_jmlr_prospective_suite.py'
    reference=load_checker(path); rng=random.Random(47)
    for _ in range(30):
        claims=[F(rng.randint(-20,20),rng.randint(1,13)) for _ in range(12)]
        responses=[[F(rng.randint(-20,20),rng.randint(1,13)) for _ in range(5)] for _ in claims]
        lipschitz=F(rng.randint(0,20),rng.randint(1,11))
        assert stable_error(claims,responses,lipschitz)==reference.stable_error(claims,responses,lipschitz)


@pytest.mark.parametrize('case',range(12))
def test_recovered_original_certificates_against_two_checkers(case):
    from check_finite_tables import load_checker
    original=load_checker(ROOT/'extensions/recovered/check_sparse_certificate.py')
    payload=json.loads((ROOT/f'extensions/recovered/generated/sparse_certificates/case_{case:02d}.json').read_text())
    c=payload['contract']; candidate=payload['candidate']; tr=payload['transcript']
    bounds={k:v for k,v in c.items() if k not in ['d','m','s','a_min']}; bounds['amin']=c['a_min']
    converted=dict(schema='tcabb.sparse.v1',d=c['d'],m=c['m'],s=c['s'],bounds=bounds,
                   observations=tr['calibration']+[row for pair in tr['source'] for row in pair],
                   gain=candidate['gains'],product=candidate['v_products'],weight=candidate['weight'])
    a=check(converted); b=original.verify(payload)
    assert a['radius']==F(b['uniform_output_error_upper'])
    assert a['kappa']==F(b['restricted_margin_lower'])
    assert a['gain_error']==F(b['gain_error_upper'])
    assert a['threshold_error']==F(b['threshold_error_upper'])
