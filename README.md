# InfoVGAE
The source code for SIGIR 2022 paper: "Unsupervised Belief Representation Learning with Information-Theoretic Variational Graph Auto-Encoders"

## Training

To run InfoVGAE on Eurovision dataset:

```
python3 main.py --config_name InfoVGAE_eurovision_3D
```

To run InfoVGAE on Election dataset:

```
python3 main.py --config_name InfoVGAE_election_3D
```

To run InfoVGAE on Voteview 105th Congress dataset:
```
python3 main.py --config_name InfoVGAE_bill_3D
```

To run InfoVGAE on TIMME dataset:
```
python3 main.py --config_name InfoVGAE_timme_3D
```

To run InfoVGAE on TIMME dataset with follow (friend) links:
```
python3 main.py --config_name InfoVGAE_timme_follow_3D
```

The `embeddings`, `labels`, `figures`, and `top-k tweets` (only applicable for Twitter datasets), etc, will be saved in `./output`

## Dataset

We uploaded the pre-processed datasets with smaller size, due to the file size limits of Github. The datasets are located in `dataset/election`, `dataset/eurovision`, and `dataset/bill`. It may takes some time to clone this repo (`297MB`). After cloning this repo, please run:

```
unzip dataset/bill/bmap2.pkl.zip; unzip dataset/bill/data_80_115.pkl.zip
```

## Evaluation

Evaluation will be automaticly triggered after the training process. To evaluate again, modify the `evaluator.init_from_dir()` in `evaluate.py`.

## Other arguments for training:

> General

`--use_cuda`: training with GPU

`--epochs`: iterations for training

`--learning_rate`: learning rate for training

`--device`: which gpu to use. empty for cpu.

`--num_process`: num process for pandas processing

> Data

`--data_path`: csv path for data file

`--stopword_path`: stopword path for text parsing

`--kthreshold`: tweet count threshold to filter not popular tweets.

`--uthreshold`: user count threshold to filter not popular users.

> For InfoVGAE model

`--hidden1_dim`: the latent space dimension of first layer

`--hidden2_dim`: the latent space dimension of target layer

> Result

`--output_path` path to save the result
