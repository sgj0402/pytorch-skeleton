# pytorch-skeleton

This repository is based on following projects.

- [Pytorch](https://pytorch.org/get-started/locally/)
- [Pytorch Ignite](https://github.com/pytorch/ignite)
- [Hydra](https://github.com/facebookresearch/hydra)
- [Aim](https://github.com/aimhubio/aim)

# Requirements

### Pytorch

refer to https://pytorch.org/get-started/locally/

### Pytorch Ignite

```
pip install pytorch-ignite
```

### Hydra
```
pip install hydra-core --upgrade
```

### Aim
```
pip3 install aim
```

# How to Use

- Implement `ignite_util.py` and `load_util.py`.
- Modify `config/config.yaml` and `config/setting/train.yaml` / `config/setting/test.yaml`.
- Run `main.py` and `aim up`.