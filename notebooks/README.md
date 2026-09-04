# LEGOde studies

This directory contains reproducible, notebook-like Python scripts. Files use
`# %%` cells, so they can be opened interactively by Jupyter-compatible editors
or executed from the command line.

Install the optional dependencies from the project root:

```bash
python -m pip install -e ".[notebooks]"
```

Run the Duffing active-Kriging comparison:

```bash
python notebooks/duffing_ak_legode.py
```

Results are written to `results/duffing` beside the study file, independent of
the directory from which Python was started. In a native `.ipynb` kernel where
`__file__` is unavailable, the current notebook-server directory is used. Edit
`OUTPUT_DIR`, `DURATION`, `POINTS`, and `MAX_BUDGET` in the final cell when a
different setup is needed.

The study writes a JSON summary, a CSV budget history, and a comparison figure.
It demonstrates:

- direct integration of an implicit LEGOde ensemble with SciPy;
- separation of the known linear LHS from the evaluated nonlinear RHS;
- active selection of exact nonlinear-restoring-force samples;
- reconstruction of a second LEGOde ensemble using a Kriging RHS term; and
- displacement error and physical-residual convergence as the sample budget grows.
