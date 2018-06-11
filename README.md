# TensorFlow Bootstrap

A bootstrap for a TensorFlow project

## Run

### Docker

Before run, `$JOB_DIR` and `$DATASET_DIR` should be set.

These can be set by writing `config.sh`, whose template is `config.sh.template`.

```bash
> bash run_docker.sh train|export|tensorboard|encode|build $ARGS 
```

### Bash shell

Before run, `$JOB_DIR`, `$LOG_DIR` and `$DATASET_DIR` should be set.

```bash
> bash run_venv.sh train|export|tensorboard|encode|build $ARGS
```
## Extension

### 1. Dataset

Define dataset spec.

Make a new file at `datasets/dataset` and inherit `tflibs.datasets.BaseDataset`.

### 2. Model

Define model fn.

Make a new file at `models/` and inherit `tflibs.model.Model`.

## TF Libs
See [tflibs](https://github.com/shygiants/tflibs).