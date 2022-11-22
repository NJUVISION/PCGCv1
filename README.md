# Learned Point Cloud Geometry Compression
___
**This project is no longer maintained, please use the updated version: https://github.com/NJUVISION/PCGCv2 and our latest work: https://github.com/NJUVISION/SparsePCGC**

Jianqiang Wang, Hao Zhu, Haojie Liu, Zhan Ma.  **[[arXiv]](https://arxiv.org/abs/1909.12037)**


<p align="center">
  <img src="figs/intro.png?raw=true" alt="introduction" width="800"/> (a)

  <img src="figs/framework.png?raw=true" alt="framework" width="800"/> (b)

 An illustrative overview in (a) and detailed diagram in (b) for point cloud geometry compression consisting of a pre-processing for PCG voxelization, scaling & partition, a compression network for compact PCG representation, and metadata signaling, and post-processing for PCG reconstruction and rendering. “Q” stands for “Quantization”, “AE” and “AD” are Arithmetic Encoder and Decoder respectively. “Conv” denotes the convolution layer with the number of the output, channels, and kernel size.
</p>

## Requirements
- ubuntu16.04
- python3
- tensorflow-gpu=1.13.1
- Pretrained models: https://box.nju.edu.cn/f/19f915f4be0643fc8862/
- ShapeNet Dataset: http://yun.nju.edu.cn/f/6d39b9cba0/
- Test data: https://box.nju.edu.cn/f/5ab2aa4dfd9941f5aaae/


## Usage 

### **Encoding**

```shell
python test.py compress "testdata/8iVFB/longdress_vox10_1300.ply" \
        --ckpt_dir="checkpoints/hyper/a6b3/"
```

### **Decoding**

```shell
python test.py decompress "compressed/longdress_vox10_1300" \
        --ckpt_dir="checkpoints/hyper/a6b3/"
```

Other default options: **--scale=1, --cube_size=64, --rho=1.0, --mode='hyper', --modelname='models.model_voxception'**

#### Examples
Please refer to `demo.ipynb` for each step.

---
### **Evaluation**
```shell
 python eval.py --input "testdata/8iVFB/longdress_vox10_1300.ply" \
                --rootdir="results/hyper/" \
                --cfgdir="results/hyper/8iVFB_vox10.ini" \
                --res=1024
```

Different parameters are required for different dataset, for example:

```shell
 python eval.py --input "testdata/Sparse/House_without_roof_00057_vox12.ply" \
                --rootdir "results/hyper/" \
                --cfgdir "results/hyper/House_without_roof_00057_vox12.ini" \
                --res=4096
```

The detailed cfgs and results can be downloaded in https://box.nju.edu.cn/f/b78aeedc0453442aafe5/
And several examples of decoded point clouds can be download in https://box.nju.edu.cn/d/f6a6f8ae61c94cea9248/

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
python train_hyper.py --alpha=0.75 \
        --prefix='hyper_' --batch_size=8 --init_ckpt_dir='checkpoints/hyper/a0.75b3' --reset_optimizer=1
```
or
```shell
python train_factorized.py --alpha=2  \
        --prefix='voxception_' --batch_size=8 --init_ckpt_dir='./checkpoints/factorized/a2b3' --reset_optimizer=1
```

## Comparison
### Objective  Comparison
`results.ipynb`
### Qualitative Evaluation
<center class="half">
  <img src="figs/redandblack.png?raw=true" height="260" alt="redandblack"/> <img src="figs/phil.png?raw=true" height="260" alt="phil"/>
</center>

## Update
- 2021.01.01 the paper was published on TCSVT. (Wang, J., Zhu, H., Liu, H., & Ma, Z. (2021). Lossy Point Cloud Geometry Compression via End-to-End Learning. IEEE Transactions on Circuits and Systems for Video Technology, 31, 4909-4923.)
- 2020.10.03 open source.
- 2020.02.20 smbmit to TCSVT.
- 2019.11.14 submit to AVS PCC (continue to study in EE).
- 2019.10.09 submit to TIP. (rejected)
- 2019.04.29 submit to BMVC-2019. (rejected)


## Issues
- Error on GPU: sometimes the point clouds may fail to decode correctly due to the randomness on GPU. 
  You can test on CPU by setting os.environ['CUDA_VISIBLE_DEVICES']="" in "evel.py".
  Or encode and decode at the same time,  see "compress_hyper" in "transform.py".


### Authors
These files are provided by Nanjing University [Vision Lab](http://vision.nju.edu.cn/). Thanks for the help from SJTU Cooperative Medianet Innovation Center. Please contact us (wangjq@smail.nju.edu.cn) if you have any questions. 
