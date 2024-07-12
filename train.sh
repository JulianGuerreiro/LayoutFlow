# Train LayoutFlow model using RICO dataset and using only 1 gpu
CUDA_VISIBLE_DEVICES=0 python3 src/train.py dataset=RICO model=LayoutFlow model.sampler.out_dim=9 trainer.check_val_every_n_epoch=25 trainer.max_epochs=2500 

### Uncomment below to train using the PubLayNet dataset 
# python3 src/train.py model=LayoutFlow dataset=PubLayNet

### To train using Diffusion instead simply replace LayoutFlow with LayoutDMx as shown below 
# python3 src/train.py model=LayoutDMx dataset=PubLayNet