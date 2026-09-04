# LEGOde studies

This directory contains reproducible studies. Keep every study self-contained
under `examples/<study-name>/` using this layout:

```text
examples/<study-name>/
├── README.md
├── scripts/     # Executable source files
├── notebooks/   # Interactive .ipynb notebooks
└── results/     # Generated data, figures, and summaries
```

The `scripts` and `notebooks` directories may contain alternative forms of the
same analysis. Code should write generated artifacts only to its study's
`results` directory so that studies do not share mutable output paths.

Install the optional dependencies from the project root:

```bash
python -m pip install -e ".[examples]"
```

Available studies:

- [`duffing`](duffing/README.md): direct integration versus active Kriging.
