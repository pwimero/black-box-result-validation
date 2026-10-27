# Black-box result validation

Executable checks, frozen numeric artifacts, observed transcripts and candidate
certificates. Python 3.11 is the recorded runtime. No training is required.

Run the full exact replay using the standard library:

```sh
python3 verify.py
```

Rerun the numerical experiments and focused tests:

```sh
python3 -m pip install -r requirements.txt
python3 replay_recovered.py
python3 -m pytest -q
```

Reports are generated in the ignored `reports/` directory. The numerical rerun
compares outputs with frozen original hashes. `MANIFEST.sha256` binds all released
inputs. The exact replay takes several minutes on a single CPU.

`extensions/` contains independent rational checkers, original and fresh
certificate fixtures, numerical generators and known-anchor checks. `legacy/`
contains independent frozen-evidence checkers and their required weights, inputs,
response tables, proof sidecars and hashed experiment protocol. `data/` contains
reference output summaries. See `VALIDATION.md` for the evidence boundary.
