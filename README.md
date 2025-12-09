<p align="center">
  <img src="https://aim-uofa.github.io/PMLoss/assets/favicon.svg" alt="Logo" style="width: 70px; height: 70px; margin-right: 20px; margin-bottom: -30px;">
  <h1 align="center">Revisiting Depth Representations for Feed-Forward 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://github.com/dc-walker">Duochao Shi*</a>
    .
    <a href="https://lhmd.top">Weijie Wang*</a>
    ·
    <a href="https://donydchen.github.io">Donny Y. Chen</a>
    ·
    <a href="https://steve-zeyu-zhang.github.io">Zeyu Zhang</a>
    ·
    <a href="https://jwbian.net/">Jia-Wang Bian</a>
    ·
    <a href="https://bohanzhuang.github.io">Bohan Zhuang</a>
    ·
    <a href="https://cshen.github.io/">Chunhua Shen</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.05327">Paper</a> | <a href="https://aim-uofa.github.io/PMLoss/">Project Page</a> | <a href="https://huggingface.co/dc-walker/PM-Loss/">Models</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://aim-uofa.github.io/PMLoss/assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>


<p align="center">
We introduce <strong>PM-Loss</strong>, a novel regularization loss based on a learned pointmap for feed-forward 3DGS, leading to more coherent 3D geometry and better rendering.
</p>

## News
- **09/06/25 Update:** Check out our <a href="https://github.com/ziplab/ZPressor">ZPressor</a>, a plug-and-play module that compresses multi-view inputs for scalable feed-forward 3DGS, enabling existing feed-forward 3DGS models to scale to over 100 input views!

## Installation

Our code is developed based on pytorch 2.4.0, CUDA 12.4 and python 3.10. 

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
git clone https://github.com/aim-uofa/PM-Loss
cd PM-Loss
conda create -y -n pmloss python=3.10
conda activate pmloss

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.27.post2
pip install git+https://github.com/facebookresearch/pytorch3d.git # For training only
pip install -r requirements.txt
```

We also provide a pip-installable package for PM-Loss. If you want to use our loss function in your own project, you can install it directly by running:
```bash
pip install git+https://github.com/aim-uofa/PM-Loss#subdirectory=pmloss # need pytorch3d installed
```

## Datasets

For our view synthesis experiments with Gaussian splatting, we primarily use the [DL3DV](https://github.com/DL3DV-10K/Dataset) dataset for training. We evaluate our model on both [DL3DV](https://github.com/DL3DV-10K/Dataset) and [RealEstate10K](https://google.github.io/realestate10k/index.html). For all experiments, we use a resolution of 256x448. Our data processing approach is adapted from previous works, including [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), and [DepthSplat](https://github.com/cvg/depthsplat).

By default, we assume the datasets are placed in `datasets/re10k` and `datasets/DL3DV_480P`. Otherwise you will need to specify your dataset path with `dataset.roots=[YOUR_DATASET_PATH]` in the running script.

### DL3DV

For the test set, we use the [DL3DV-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark) split, which contains 140 scenes for evaluation. For the training set, we use the [DL3DV-480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset.

We provide a script, `src/scripts/convert_dl3dv.py`, to process both the training and test sets. Running this script will convert the original data into the required format. Please note that you will need to update the dataset paths in the aforementioned processing scripts.

### RealEstate10K

Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p dataset (360x640 resolution), which can be used directly in our codebase.

## Running the Code

We release the version of PM-Loss based on [DepthSplat](https://github.com/cvg/depthsplat) architecture, specifically the version released in October 2024.

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

- get our [pretrained models](https://huggingface.co/dc-walker/PM-Loss/tree/main), and save them to `checkpoints/`

- run the following:

```bash
# dl3dv
python -m src.main \
+experiment=dl3dv mode=test \
dataset/view_sampler=evaluation \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_bound_aware.json \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
model.encoder.offset_mode=unconstrained \
checkpointing.pretrained_model=checkpoints/pmloss.ckpt \
test.compute_scores=true

# re10k
python -m src.main \
+experiment=re10k mode=test \
dataset/view_sampler=evaluation \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/re10k_bound_aware.json \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
model.encoder.offset_mode=unconstrained \
checkpointing.pretrained_model=checkpoints/pmloss.ckpt \
test.compute_scores=true
```

### Training

- Download [pretrained weights of DepthSplat](https://huggingface.co/dc-walker/PM-Loss/tree/main/depthsplat_pretrained), and save them to `depthsplat_pretrained/` 

- Run the following:
```bash
python -m src.main +experiment=dl3dv data_loader.train.batch_size=1 \
model.encoder.offset_mode=unconstrained \
loss=[mse,lpips,pcd] \
loss.pcd.weight=0.005 loss.pcd.gt_mode=vggt loss.pcd.ignore_large_loss=100.0 \
dataset.image_shape=[256,448] \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
dataset.view_sampler.min_distance_between_context_views=20 \
dataset.view_sampler.max_distance_between_context_views=50 \
trainer.max_steps=100001
```

## Citation
If you find our work useful for your research, please consider citing us:

```bibtex
@article{shi2025pmloss,
  title={Revisiting Depth Representations for Feed-Forward 3D Gaussian Splatting},
  author={Shi, Duochao and Wang, Weijie and Chen, Donny Y. and Zhang, Zeyu and Bian, Jiawang and Zhuang, Bohan and Shen, Chunhua},
  journal={arXiv preprint arXiv:2506.05327},
  year={2025}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at 
dcshi@zju.edu.cn

## Acknowledgements

This project is developed with several fantastic repos: [VGGT](https://github.com/facebookresearch/vggt), [MVSplat](https://github.com/donydchen/mvsplat) and [DepthSplat](https://github.com/cvg/depthsplat). We thank the original authors for their excellent work.
