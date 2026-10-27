"""Replay independent certificates and cross-check separate producer evidence."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json

from check_sparse_certificate import verify


ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT/'generated/frontier_verification.json').read_text())
truth_records = records['sparse_readout_extension']['noise_records']
cases = sorted((ROOT/'generated/sparse_certificates').glob('case_*.json'))
assert len(cases) == 12
accepted = []
for case, truth_record in zip(cases, truth_records):
    answer = verify(json.loads(case.read_text()))
    assert answer['status'] == 'ACCEPT'
    # This comparison is external validation. Ground truth is never supplied
    # to the observation-only checker itself.
    actual = Fraction(truth_record['maximum_actual_error'])
    radius = Fraction(answer['uniform_output_error_upper'])
    assert actual <= radius
    accepted.append({'certificate':case.name, 'checker':answer,
                     'largest_tested_actual_error':str(actual),
                     'all_tested_interventions_within_checker_radius':True})

base = json.loads(cases[1].read_text())
mutations = []
for name in ['gain','noise','weight','geometry','product']:
    altered = deepcopy(base)
    if name == 'gain':
        altered['candidate']['gains'][0][0] = '1/5'
    elif name == 'noise':
        altered['contract']['eta'] = '0'
    elif name == 'weight':
        altered['candidate']['weight'][0] = '0'
    elif name == 'geometry':
        altered['contract']['K'] = '0'
    else:
        i,k = next((i,k) for i in range(4) for k in range(4)
                   if i != k and Fraction(altered['candidate']['gains'][i][k]) == 0)
        altered['candidate']['v_products'][i][k] = '1/10'
    try:
        verify(altered)
    except ValueError as error:
        mutations.append({'mutation':name,'status':'REJECT','reason':str(error)})
    else:
        raise AssertionError('altered certificate was accepted: '+name)

result = {'status':'PASS','accepted_cases':len(accepted),
          'tested_interventions':240, 'rejected_mutations':mutations,
          'cases':accepted}
(ROOT/'generated/sparse_checker_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'cases'},indent=2))
