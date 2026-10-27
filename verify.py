"""Replay the frozen validation evidence and check its checksums."""
from pathlib import Path
import subprocess,sys,hashlib,json,os
ROOT=Path(__file__).resolve().parent


def run(args,cwd=ROOT,report=None):
    print('Running:', ' '.join(map(str,args)),flush=True)
    result=subprocess.run([sys.executable,*map(str,args)],cwd=cwd,text=True,capture_output=True)
    if report: (ROOT/'reports'/report).write_text(result.stdout)
    if result.returncode:
        print(result.stdout); print(result.stderr,file=sys.stderr)
        raise SystemExit(result.returncode)


def verify_manifest():
    manifest=ROOT/'MANIFEST.sha256'
    if not manifest.exists(): raise SystemExit('Missing manifest')
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Manifest mismatch: '+name)
    print('Release manifest matches.',flush=True)


if __name__=='__main__':
    (ROOT/'reports').mkdir(exist_ok=True)
    verify_manifest()
    run(['extensions/reconstruct.py'])
    run(['extensions/recovered/replay_certificates.py'],report='recovered_sparse_replay.json')
    run(['extensions/check_known_anchor.py'])
    run(['reference_checker/check_continuous_triangle.py'],ROOT/'legacy','continuous_replay.json')
    run(['reference_checker/check_jmlr_defensibility_suite.py'],ROOT/'legacy','retrospective_replay.json')
    run(['extensions/check_finite_tables.py'],report='prospective_execution.txt')
    verify_manifest()
    print('PASS: exact sparse reconstruction, known anchors, continuum and finite-table replays.')
