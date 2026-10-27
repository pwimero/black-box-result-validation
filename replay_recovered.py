"""Replay recovered original extension experiments and check byte-level reproduction."""
from pathlib import Path
import subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
STEPS=[('generate_extensions.py','recovered_generator.json'),
       ('replay_certificates.py','recovered_sparse_replay.json'),
       ('check_wellposedness.py','recovered_wellposedness.json'),
       ('check_softplus.py','recovered_softplus.json')]
if __name__=='__main__':
    (ROOT/'reports').mkdir(exist_ok=True)
    recovered=ROOT/'extensions/recovered'
    for source,report in STEPS:
        print('Running:',source,flush=True)
        result=subprocess.run([sys.executable,str(recovered/source)],text=True,capture_output=True)
        (ROOT/'reports'/report).write_text(result.stdout)
        if result.returncode:
            print(result.stderr,file=sys.stderr); raise SystemExit(result.returncode)
    original_hashes=json.loads((recovered/'ORIGINAL_SHA256.json').read_text())
    changed=[name for name,digest in original_hashes.items()
             if hashlib.sha256((recovered/name).read_bytes()).hexdigest()!=digest]
    report=dict(status='PASS' if not changed else 'DIFFERS_FROM_ARCHIVE',changed_files_after_replay=changed,
                generated_files_byte_identical=not any(n.startswith('generated/') for n in changed),
                source_files_byte_identical=not any(n.endswith('.py') for n in changed),
                supplied_summary_byte_matches={n:(recovered/'generated'/n).read_bytes()==(ROOT/'data'/n).read_bytes()
                  for n in ['frontier_verification.json','sparse_checker_results.json']})
    (ROOT/'reports/recovered_replay_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    if changed or not all(report['supplied_summary_byte_matches'].values()): raise SystemExit('Replay differs; inspect report.')
    print('PASS: all original extension outputs reproduce byte for byte.')
