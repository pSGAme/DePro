# DePro:Domain Ensemble using Decoupled Prompts for Universal Cross-Domain Retrieval [SIGIR 2025]

> Official github repository for **DePro:Domain Ensemble using Decoupled Prompts for Universal Cross-Domain Retrieval**. 
> 
> Please refer to [paper link](https://arxiv.org/abs/2312.12478) for detailed information.

## Requirements

- **Ubuntu**  22.04.4
- **Nvidia GeForce RTX4090D-24GB**
- **CUDA** 12.4

Supports for other platforms and hardwares are possible with no warrant. 

## Setups

1. Clone this repository：

``` bash
git clone https://github.com/pSGAme/DePro.git
```
or download the zip file.

2. Install dependencies：

```bash
cd ./DePro
conda env create -f DePro.yaml
conda activate DePro
```

## Data Preparation

1. Download DomainNet using scripts in `./DePro/src/data/downloads`.

   ``` bash
   cd ./src/data/downloads
   bash download_domainnet.sh
   ```
2. For Sketchy and TU-berlin, please refer to this [issue](https://github.com/kaipengfang/ProS/issues/3).

3. The directory is expected to be in the structure below:

   ```python
   ├── DomainNet
   │   ├── clipart # images from clipart domain
   │   ├── clipart_test.txt # class names for testing
   │   ├── clipart_train.txt # class names for training
   │   ├── down.sh
   │   ├── infograph
   │   ├── infograph_test.txt
   │   ├── infograph_train.txt
   │   ├── painting
   │   ├── painting_test.txt
   │   ├── painting_train.txt
   │   ├── quickdraw
   │   ├── quickdraw_test.txt
   │   ├── quickdraw_train.txt
   │   ├── real
   │   ├── real_test.txt
   │   ├── real_train.txt
   │   ├── sketch
   │   ├── sketch_test.txt
   │   └── sketch_train.txt
   ├── Sketchy
   │   ├── extended_photo
   │   ├── photo
   │   ├── sketch
   │   └── zeroshot1
   └── TUBerlin
       ├── images
       └── sketches
   ```

## Experiments

### baseline:

```bash
cd ./src/algos/depro

sh baseline.sh
```

### baseline+CPs:

```bash
cd ./src/algos/depro

sh baseline_CPs.sh
```

### depro_basic:

```bash
cd ./src/algos/depro
sh depro.sh
```

### depro_ln:

```bash
cd ./src/algos/depro
sh depro_ln.sh
```

### depro_ln_triplet_loss (all):

```bash
cd ./src/algos/depro
sh depro_ln_tri.sh
```


## Acknowledgements

Our code implementation is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [ProS](https://github.com/kaipengfang/ProS).
We thank all the authors for releasing their code.
