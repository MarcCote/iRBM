# iRBM
Infinite Restricted Boltzmann Machine

Paper on [arxiv](http://arxiv.org/abs/1502.02476) and at [ICML2015 - Deep Learning Workshop](https://sites.google.com/site/deeplearning2015/accepted-papers).

## Dependencies:
- python == 2.7
- numpy >= 1.7
- scipy >= 0.11
- theano >= 0.6

## Usage
Experiments are saved in : *`./experiments/{experiment_name}/`*.

Datasets will be downloaded and saved in : *`./datasets/{dataset_name}/`*.


### Train
See `python train_model.py --help`

**Binarized MNIST**
Training a model (infinite RBM) on binarized MNIST.
```
python train_model.py --name "best_irbm_mnist" --max-epoch 100 --batch-size 64 --ADAGRAD 0.03 irbm binarized_mnist --beta 1.01 --PCD --cdk 10
```

**CalTech101 Silhouettes**
Training a model (infinite RBM) on CalTech101 Shilhouettes.
```
python train_model.py --name "best_irbm_caltech101" --max-epoch 1000 --batch-size 64 --ADAGRAD 0.03 irbm caltech_silhouettes28 --beta 1.01 --PCD --cdk 10
```


### Evaluate
See `python eval_model.py --help`

**Binarized MNIST**
Evaluating a model trained on binarized MNIST (assuming the one above).
```
python eval_model.py experiments/best_irbm_mnist/
```

**CalTech101 Silhouettes**
Evaluating a model trained on CalTech101 Silhouettes (assuming the one above).
```
python eval_model.py experiments/best_irbm_caltech101/
```


### Sample
See `python sample_model.py --help`

**Binarized MNIST**
Generating 16 binarized MNIST digits images sampled from a trained model (assuming the one above).
```
python -u sample_model.py experiments/best_irbm_mnist/ --nb-samples 16 --view
```

**CalTech101 Silhouettes**
Generating 16 silhouette images sampled from a trained model (assuming the one above).
```
python -u sample_model.py experiments/best_irbm_caltech101/ --nb-samples 16 --view
```


### Visualize filters
See `python show_filters.py --help`

**Binarized MNIST**
Visualizing filters of a model trained on binarized MNIST (assuming the one above).
```
python show_filters.py experiments/best_irbm_mnist/
```

**CalTech101 Silhouettes**
Visualizing filters of a model trained on CalTech101 Silhouettes (assuming the one above).
```
python show_filters.py experiments/best_irbm_caltech101/
```


## Datasets
The datasets are automatically downloaded and processed. Available datasets are:
- binarized MNIST
- CalTech101 Silhouettes (28x28 pixels)


## Troubleshooting
- **I got a weird cannot convert int to float error. ``TypeError: Cannot convert Type TensorType(float32, matrix) (of Variable Subtensor{int64:int64:}.0) into Type TensorType(float64, matrix)``**

Have you [configured theano](http://deeplearning.net/software/theano/library/config.html#envvar-THEANORC)?
Here is my .theanorc config (use cpu if you do not have a CUDA capable gpu):
```
[global]
device = gpu
floatX = float32
exception_verbosity=high

[nvcc]
fastmath = True
```


- **I got an IO error about `status.json`. ``IOError: [Errno 2] No such file or directory: './experiments/.../status.json'``**

There is no `status.json` file, so it is impossible to resume the experiment. Use --force to restart the experiment form scratch.