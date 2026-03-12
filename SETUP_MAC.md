# macOS (Apple Silicon) environment setup

This repository includes a Linux-locked environment file (`environment.yml`) that cannot be solved on macOS.

Use the macOS file instead:

```bash
conda env create -f environment.macos-arm64.yml
conda activate pt12-mac
```

Install PyG extension wheels required by `pygcl`:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

Quick smoke test:

```bash
python -c "import torch, dgl; import GCL.augmentors as A; print(torch.__version__, dgl.__version__)"
python GCLink_main_SimSiam.py -h
```

Example run:

```bash
python GCLink_main_SimSiam.py -cell_type mESC -sample sample1 -Type edge_mlp
```