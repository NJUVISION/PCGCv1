# Learned Point Cloud Geometry Compression
___
Jianqiang Wang, Hao Zhu, Haojie Liu, Zhan Ma.  **[[arXiv]](https://arxiv.org/abs/1909.12037)**
<p align="center">
  <img src="figs/intro.png?raw=true" alt="introduction" width="800"/> (a)

  <img src="figs/framework.png?raw=true" alt="framework" width="800"/> (b)

 An illustrative overview in (a) and detailed diagram in (b) for point cloud geometry compression consisting of a pre-processing for PCG voxelization, scaling & partition, a compression network for compact PCG representation, and metadata signaling, and post-processing for PCG reconstruction and rendering. “Q” stands for “Quantization”, “AE” and “AD” are Arithmetic Encoder and Decoder respectively. “Conv” denotes the convolution layer with the number of the output, channels, and kernel size.

</p>



## Requirements
- ubuntu16.04
-  python3
- tensorflow-gpu=1.13.1
- Pretrained models: http://yun.nju.edu.cn/f/d8f82261c0/
- ShapeNet Dataset: http://yun.nju.edu.cn/f/6d39b9cba0/
- Test data: http://yun.nju.edu.cn/f/d834ac0be4/

## Usage 

### **Encoding**

```shell
python mycodec_factorized.py compress 'testdata/8iVFB/loot_vox10_1200.ply' -ckpt_dir='checkpoints/factorized/a2b3/' --gpu=1
```
or
```shell
python mycodec_hyper.py compress 'testdata/8iVFB/longdress_vox10_1300.ply' --ckpt_dir='checkpoints/hyper/a0.75b3/' --gpu=1
```
---

### **Decoding**

```shell
python mycodec_factorized.py decompress 'compressed/loot_vox10_1200' --ckpt_dir='checkpoints/factorized/a2b3/' --gpu=1
```
or
```shell
python mycodec_hyper.py decompress 'compressed/longdress_vox10_1300' --ckpt_dir='checkpoints/hyper/a0.75b3/' --gpu=1
```

You can also define other options including **output, model, scale, cube_size, gpu, rho**, etc. via the command line.  

#### Examples
Please refer to `test_factorized.ipynb` and `test_hyper.ipynb` for each step. 

---
### **Testing**

The testing on `testdata/` are shown in `eval_factorized.ipynb` and `eval_hyper.ipynb`.
Or run:

```shell
python eval_factorized_seqs.py
python eval_hyper_seqs.py
```
The results can be downloaded in http://yun.nju.edu.cn/f/40578f25c6/ and http://yun.nju.edu.cn/f/1a6426ffba/

---

### Training
#### **Generating training dataset**
sampling points from meshes, here we use **pyntcloud** (pip install pyntcloud)

```shell
cd dataprocess
python mesh2pc.py
```
The output point clouds can be download in http://yun.nju.edu.cn/d/227493a5bd/
```shell
python generate_dataset.py
```
the output training dataset can be download in http://yun.nju.edu.cn/d/604927e275/

--- 
#### **Training**

```shell
python mytrain_factorized.py --alpha=3
```
```shell
python mytrain_hyper.py --alpha=3
```

You can set **alpha**, **beta**, etc to adjust loss function, and provide a pretrained model via **init_ckpt_dir**.

## Comparison
### Objective  Comparison
`results.ipynb`
### Qualitative Evaluation
<center class="half">
  <img src="figs/redandblack.png?raw=true" height="260" alt="redandblack"/> <img src="figs/phil.png?raw=true" height="260" alt="phil"/>
</center>

## Update
- 2019.10.09 initial smbmission.
- 2019.10.22 submit demos, several pretrained models and training datasets.
- 2019.10.27 submit all pretrained models and evaluate on 8i voxelized full bodies.
- 2019.11.14 check bug and start testing on AVS PCC Cat3.
- 2019.11.15 test point cloud sequences using avs metric tool. (thanks for the help from Wei Yan)
- 2019.11.19 finish testing AVS Cat3.
- 2019.11.20 test AVS Cat2.
- 2019.11.26 test AVS single frame.
- 2019.11.31 doc.
- 2020.06.27 python3 & clean up.
- 2020.10.03 open source.

## Todo
- pytorch version & tensorflow2.0 version.
- training again.

### Authors
These files are provided by Nanjing University [Vision Lab](http://vision.nju.edu.cn/). And thanks for the help from SJTU Cooperative Medianet Innovation Center. Please contact us (wangjq@smail.nju.edu.cn) if you have any questions. 
