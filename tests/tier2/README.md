# Tier-2 independent baseline

`tools/tier2/generate_signals.py` creates deterministic capture files using
NumPy and the external Komm convolutional encoder. It has no imports from the
application, project FEC/interleaver modules, tests, training data, or model
artifacts. The runner invokes only the public `Analyzer.analyze_file(...)` path
for the application result.

Run the focused harness checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tier2/test_tier2_harness.py -q
```

Run a timestamped baseline with:

```powershell
.\.venv\Scripts\python.exe tools\tier2\run_tier2.py --run-dir reports\tier2\baseline\<run-id>
```

The runner continues through all configured captures after an individual
failure. It stores each capture, ground truth, full `AnalysisResult`,
comparison, and log under the run directory. The aggregate summary files are
written under `reports/tier2/baseline/`. Do not reuse a run directory or replace
an existing summary from an earlier validation. The runner refuses a nonempty
run directory or existing aggregate summary path. For a later baseline, pass a
new `--run-dir` and a new `--reports-root` directory, for example:

```powershell
.\.venv\Scripts\python.exe tools\tier2\run_tier2.py `
  --run-dir reports\tier2\baseline\<new-run-id> `
  --reports-root reports\tier2\baseline-summaries\<new-run-id>
```

GNU Radio was not installed in the baseline environment. These results must
therefore be described as NumPy/Komm independent-reference captures, not as
GNU Radio captures. The optional LDPC case is blocked unless an independent
reference encoder is separately established.
