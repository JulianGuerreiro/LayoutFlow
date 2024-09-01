# LayoutFlow: Flow Matching for Layout Generation (ECCV 2024)
This repository is the official implementation of the ECCV 2024 paper "LayoutFlow: Flow Matching for Layout Generation" ([project page](https://julianguerreiro.github.io/layoutflow/) | [paper](https://arxiv.org/pdf/2403.18187)).

## Requirements
We used the following environement for the experiments:
- Python 3.8
- CUDA 12.4
- Pytorch 2.2.2

Other dependencies can be installed using pip as follows:
```
pip install -r requirements.txt
```

## Overview
The code uses the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework and manages configurations with [Hydra](https://hydra.cc). For logging during training, we used [Weights and Biases](https://www.wandb.ai) but alternatively tensorboard can also be used by changing the logger in `conf/train.yaml`.

### Configs
The configuration files are defined in the `.yaml` found in the `conf` folder and contain hyperparameters and other settings. The values can be changed in the `.yaml` files directly (which we only recommend for data paths or permanent changes) or, alternatively, can be overidden as a command line instruction. For example changing the batch size used during training can be done like this:  
```
python src/train.py dataset=PubLayNet model=LayoutFlow dataset.batch_size=1024
```

### Models
We provide two different generative models in `src/models`, namely our flow-based approach called `LayoutFlow` and a diffusion-based approach `LayoutDMx`. The main difference between both models is just the training procedure (diffusion vs. flow). The same backbone architecture in `src/models/backbone` can be chosen for either one of them.  

### Dataset
We trained our model on the RICO and PubLayNet dataset using the dataset split reported in [LayoutFormer++](https://arxiv.org/abs/2208.08037) and [LayoutDiffusion](https://arxiv.org/abs/2303.17189). Please download the following files from this [Hugging Face repository](https://huggingface.co/JulianGuerreiro/LayoutFlow) using (make sure you have installed git lfs, otherwise the large files will not be downloaded)
```
git clone https://huggingface.co/JulianGuerreiro/LayoutFlow
```
**Important:**

You can store the data in a directory of your choosing, but you will need to add the *datapath* in the dataset config files. Specifically, change the `data_path` attribute in `conf/dataset/PubLayNet.yaml` and `conf/dataset/RICO.yaml` to the path where the respective folders are located.

*We also provide the PubLayNet split in [LayoutDM (Inoue et al.)](https://github.com/CyberAgentAILab/layout-dm), which we used for comparison with other models as described in the Appendix (Section: Results Using Different Data Split).*

### Pretrained Models
The pre-trained models can be downloaded from the [Hugging Face repository](https://huggingface.co/JulianGuerreiro/LayoutFlow) as described above. They can be used to evaluate the model or even continue training.
Note that the `.tar` files in `pretrained` are used for the FID model and are identical to the ones used in [LayoutDiffusion](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion/eval_src/net) and do not need to be downloaded seperately. Furthermore, the `.pt` files are additionally used for the FID calculation. 

## Testing
A model can be evaluated on various tasks by calculating FID, Alignment, Overlap and mIoU. The example below shows a minimal example: 
```
python3 src/test.py model=[MODEL] dataset=[DATASET] task=[TASK] cond_mask=[MASK] checkpoint=[DIR_TO_CHECKPOINT]
```
- [MODEL]: either `LayoutFlow` or `LayoutDMx`
- [DATASET]: either `PubLayNet` or `RICO`
- [TASK]:
    - `uncond` Unconditional Generation (Layout is generated completely from scratch)
    - `cat_cond` Category-conditioned Generation (Categories are given, bounding boxes are predicted)
    - `size_cond` Categoy-and-Size-conditioned Generation (Categories and size of the bounding boxes are given, position of bounding boxes is predictec)
    - `elem_compl` Element Completion (Predicts new bounding boxes, based on unfinished layout)
    - `refinement` Refinement (Refines a slightly noisy layout)
- [MASK]: Same options as `TASK`, but without `refinement`. This describes which conditioning mask is applied. Select the same option as in `TASK`, except for `refinement`, in that case, please use the `uncond` mask.
- [DIR_TO_CHECKPOINT]: Directory to the model weigths

**Other useful settings (see also `test.yaml` config file)**
- `model.inference_steps` (Default: 100): Number of steps used to solve the ODE (e.g. can be reduced to 50 with basically same performance for LayoutFlow)
- `calc_miou` (Default: `False`): Whether to calculate the mIoU (can take some time, especially with PubLayNet)
- `multirun` (Default: `False`): Whether to generate layouts multiple times, to increase confidence of the score (runs 10 times and then averages)
- `visualize` (Default: `False`): Whether to visualize some of the created layouts (make sure to make a folder called `vis` for the images to be saved in)

The results will be saved in the `results` directory as a `.pt` file. To re-evaluate the files, you can set the variable `load_bbox` to the path of the `.pt` file. 

**Note**

*Since the generation task is non-deterministic, there will be some variations in the results and it will not match the values of the paper perfectly. The provided weights are also not the original weights we used in the paper, as we re-trained the model after refactoring. Nonetheless, we evaluated the newly trained models and they were very close to the reported values after using `multirun`.*

## Training
For training, we provide the `train.sh` file, where you can comment out the model that you would like to train. If you want to train the model with different hyperparameters, you can change the values in the `.sh` file, for example add `model.optimizer.learning_rate=0.0001` to change the learning rate.

We recommend using a single GPU for training as that has shown the best results under the current hyperparameters.

**Useful settings**
- `model.optimizer.lr` (default: 0.0005): learning rate
- `model.cond` (default: random4): conditioning masked used during training, random4 samples the proposed 4 conditioning masks randomly
- `model.sampler.distribution` (default: gaussian): initial distribution (e.g. `uniform` or `gauss_uniform`) 
- `model.train_traj` (default: linear): training trajectory, alternative options are `sincos` or `sin`
- `model.add_loss_weight` (default: 0.2): weighting of additional geometrical loss  
*For more settings check out the `.yaml` files*

## Citation
If this work is helpful for your research, please cite our paper:
```
@article{guerreiro2024layoutflow,
  title={LayoutFlow: Flow Matching for Layout Generation},
  author={Guerreiro, Julian Jorge Andrade and Inoue, Naoto and Masui, Kento and Otani, Mayu and Nakayama, Hideki},
  journal={arXiv preprint arXiv:2403.18187},
  year={2024}
}
```

### Acknowledgments
We want to acknowledge that some parts of our code (mainly some utils functions for the evaluation) are based on code used in the following projects: [LayoutDiffusion](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion) and [LayoutDM](https://github.com/CyberAgentAILab/layout-dm?tab=readme-ov-file).
