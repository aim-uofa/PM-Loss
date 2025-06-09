<p align="center">
  <img src="assets/favicon.svg" alt="Logo" style="width: 70px; height: 70px; margin-right: 20px; margin-bottom: -30px;">
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
  <h3 align="center"><a href="https://arxiv.org/abs/2506.05327">Paper</a> | <a href="https://aim-uofa.github.io/PMLoss/">Project Page</a> | <a href="https://aim-uofa.github.io/PMLoss/">Models</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="/assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>


<p align="center">
We introduce <strong>PM-Loss</strong>, a novel regularization loss based on a learned point map for feed-forward 3DGS,  leading to smoother 3D geometry and better rendering.
</p>

## News
- **09/06/25 Update:** Check out our <a href="https://github.com/ziplab/ZPressor">ZPressor</a>, a plug-and-play module that compresses multi-view inputs for scalable feed-forward 3DGS, enabling existing feed-forward 3DGS models to scale to over 100 input views!

## Method
<p align="center">
  <a href="">
    <img src="/assets/pipeline.png" alt="Logo" width="100%">
  </a>
</p>
<strong>Overview of our PM-Loss</strong>.
The process begins by estimating a dense point map of the scene using a pre-trained model. 
This estimated point map then serves as direct 3D supervision for training a feed-forward 3D Gaussian Splatting model. 
Crucially, unlike conventional methods relying predominantly on 2D supervision, our approach leverages explicit 3D geometric cues, leading to enhanced 3D shape fidelity.


## TODOs
- [ ] Release Code.
- [ ] Release Model Checkpoints.

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
