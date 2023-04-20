# CLARA

## Installation

### Requirements

- Linux with Python ≥ 3.6
- [PyTorch ≥ 1.4.0](https://pytorch.org/)
- [0.5 > DGL ≥ 0.4.3](https://www.dgl.ai/pages/start.html)
- `pip install -r requirements.txt`
- Install [RDKit](https://www.rdkit.org/docs/Install.html) with `conda install -c conda-forge rdkit=2019.09.2`.

## Quick Start

1. 运行prepare.main方法进行数据准备工作。完成数据读取、异构图构建以及数据增强

2. 运行CLARA代码
```bash
cd CLARA
sh run_loop.sh 
```

[comment]: <> (```bash)

[comment]: <> (# run comparative experiment)

[comment]: <> (sh run_loop.sh)

[comment]: <> (```)
