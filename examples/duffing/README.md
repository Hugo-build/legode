# Duffing active-Kriging study

This study compares direct integration of a Duffing oscillator with an active
Kriging surrogate built using LEGOde.

From the project root, install the study dependencies and run the script:

```bash
python -m pip install -e ".[examples]"
python examples/duffing/duffing_ak_legode.py
```

The script is organized into `# %%` windows that can be run one at a time in
Jupyter-compatible editors. Each stage prints its result immediately, and the
three plotting windows show and save the solution comparison, learned nonlinear
force, and budget convergence separately. Numeric CSV and JSON outputs and the
figures are written to `examples/duffing/results`, independent of the directory
from which the script is started. Edit `OUTPUT_DIR`, `DURATION`, `POINTS`, and
`MAX_BUDGET` in the configuration window to change the setup.

The study demonstrates:

- direct integration of an implicit LEGOde ensemble with SciPy;
- separation of the known linear LHS from the evaluated nonlinear RHS;
- active selection of exact nonlinear-restoring-force samples;
- reconstruction of a second LEGOde ensemble using a Kriging RHS term; and
- displacement error and physical-residual convergence as the sample budget grows.
