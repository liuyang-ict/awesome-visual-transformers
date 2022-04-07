# <p align=center> [A Survey of Visual Transformers](https://arxiv.org/abs/2111.06091)</p>

  
##### <p align=center> [Yang Liu](https://scholar.google.com/citations?user=ock4qjYAAAAJ&hl=zh-CN), [Yao Zhang](https://scholar.google.com/citations?user=vxfJSJIAAAAJ&hl=zh-CN), [Yixin Wang](https://scholar.google.com/citations?user=ykYrXtAAAAAJ&hl=zh-CN), [Feng Hou](https://scholar.google.com/citations?user=gp-OCDoAAAAJ&hl=zh-CN), [Jin Yuan](https://scholar.google.com/citations?hl=zh-CN&user=S1JGPCMAAAAJ), [Jiang Tian](https://scholar.google.com/citations?user=CC_HnVQAAAAJ&hl=zh-CN), [Yang Zhang](https://scholar.google.com/citations?user=fwg2QysAAAAJ&hl=zh-CN), [Zhongchao Shi](https://scholar.google.com/citations?hl=zh-CN&user=GASgQxEAAAAJ), [JianPing Fan](https://scholar.google.com/citations?user=-YsOqQcAAAAJ&hl=zh-CN), [Zhiqiang He](https://ieeexplore.ieee.org/author/37085386255)</p>



# <p align=center>`Awesome Visual Transformers`</p>

A curated list of awesome Transformers resources in medical imaging (**in chronological order**), inspired by the other awesome-initiatives. We intend to regularly update the relevant latest papers and their open-source implementations on this page. If you find some overlooked papers, please open an issue or contact at liuyang20c@mails.ucas.ac.cn.

## Overview
- [Original Transformer](#original-transformer)
- [Transformer for Classification](#transformer-for-classification)
- [Transformer for Detection](#transformer-for-detection)
- [Transformer for Segmentation](#transformer-for-segmentation)
- [Transformer for 3D Visual Recognition](#transformer-for-3d-visual-recognition)
- [Transformer for Multi-Sensory Data Stream](#transformer-for-multi-sensory-data-stream)
- [Theoretical](#theoretical)


# Original Transformer
**Attention Is All You Need.** [12th Jun. 2017] [NeurIPS, 2017].<br>
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.*<br>
 [[PDF](https://arxiv.org/abs/1706.03762)] [[Github](https://github.com/tensorflow/tensor2tensor)]

# Transformer for Classification

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** [10th Mar. 2021] [ICLR, 2021].<br>
*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.*<br>
 [[PDF](https://arxiv.org/abs/2010.11929)] [[Github](https://github.com/google-research/vision_transformer)]
 
**Visual Transformers: Token-based Image Representation and Processing for Computer Vision.** [5th Jun 2020].<br>
*Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Zhicheng Yan, Masayoshi Tomizuka, Joseph Gonzalez, Kurt Keutzer, Peter Vajda.*<br>
 [[PDF](https://arxiv.org/abs/2006.03677)] 
 
**Bottleneck Transformers for Visual Recognition.** [2nd Aug. 2021] [CVPR, 2021].<br>
*Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, Ashish Vaswani.*<br>
 [[PDF](https://arxiv.org/abs/2101.11605)] [[Github](https://github.com/rwightman/pytorch-image-models)]
 
**Training data-efficient image transformers & distillation through attention.** [15th Jan. 2021] [ICML, 2021].<br>
*Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.*<br>
 [[PDF](https://arxiv.org/abs/2012.12877)]  [[Github](https://github.com/facebookresearch/deit)]
 
**ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases** [10th Jun. 2021] [ICLR, 2021].<br>
*Christos Matsoukas, Johan Fredin Haslum, Magnus Söderberg, Kevin Smith.*<br>
 [[PDF](https://arxiv.org/abs/2103.10697)] [[Github](https://github.com/facebookresearch/convit)]
 
**Incorporating Convolution Designs into Visual Transformers** [20th Apr. 2021] [ICCV, 2021].<br>
*Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, Wei Wu.*<br>
 [[PDF](https://arxiv.org/abs/2103.11816)] [[Github](https://github.com/rishikksh20/CeiT-pytorch)]
 
**LocalViT: Bringing Locality to Vision Transformers.** [12nd Apr. 2021].<br>
*Yawei Li, Kai Zhang, JieZhang Cao, Radu Timofte, Luc van Gool.*<br>
 [[PDF](https://arxiv.org/abs/2104.05707)] [[Github](https://github.com/ofsoundof/LocalViT)]
 
**Conditional Positional Encodings for Vision Transformers.** [22nd Feb. 2021].<br>
*Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, Xiaolin Wei, Huaxia Xia, Chunhua Shen.*<br>
 [[PDF](https://arxiv.org/abs/2102.10882)] [[Github](https://github.com/Meituan-AutoML/CPVT)]
 
**ResT: An Efficient Transformer for Visual Recognition.** [14th Oct. 2021] [NeurIPS, 2021].<br>
*Qinglong Zhang, YuBin Yang.*<br>
 [[PDF](https://arxiv.org/abs/2105.13677)] [[Github](https://github.com/wofmanaf/ResT)]
 
**Early Convolutions Help Transformers See Better.** [25th Oct. 2021] [NeurIPS, 2021 ].<br>
*Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollár, Ross Girshick.*<br>
 [[PDF](https://arxiv.org/abs/2106.14881)] [[Github](https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch)]
 
**CoAtNet: Marrying Convolution and Attention for All Data Sizes.** [15th Sep. 2021] [NeurIPS, 2021].<br>
*Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan.*<br>
 [[PDF](https://arxiv.org/abs/2106.04803)] [[Github](https://github.com/chinhsuanwu/coatnet-pytorch)]
 
**Scaling Local Self-Attention for Parameter Efficient Visual Backbones.** [7th Jun. 2021] [CVPR, 2021].<br>
*Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, Jonathon Shlens.*<br>
 [[PDF](https://arxiv.org/abs/2103.12731)] [[Github](https://github.com/rwightman/pytorch-image-models)]
 
**Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** [17th Aug. 2021] [ICCV, 2021].<br>
*Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.*<br>
 [[PDF](https://arxiv.org/abs/2103.14030)] [[Github](https://github.com/microsoft/Swin-Transformer)]

**VOLO: Vision Outlooker for Visual Recognition.** [24th Jun. 2021].<br>
*Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, Shuicheng Yan.*<br>
 [[PDF](https://arxiv.org/abs/2106.13112)] [[Github](https://github.com/sail-sg/volo)]
 
**Transformer in Transformer.** [26th Oct. 2021] [NeurIPS, 2021].<br>
*Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang.*<br>
 [[PDF](https://arxiv.org/abs/2103.00112)] [[Github](https://github.com/huawei-noah/CV-backbones)]
 
**Twins: Revisiting the Design of Spatial Attention in Vision Transformers.** [30th Sep. 2021] [NeurIPS, 2021].<br>
*Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, Chunhua Shen.*<br>
 [[PDF](https://arxiv.org/abs/2104.13840)] [[Github](https://github.com/Meituan-AutoML/Twins)]
 
 **Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding.** [27th May 2021] [ICCV, 2021].<br>
*Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, Jianfeng Gao.*<br>
 [[PDF](https://arxiv.org/abs/2103.15358)] [[Github](https://github.com/microsoft/vision-longformer)]
 
**Focal Self-attention for Local-Global Interactions in Vision Transformers.** [1st Jul. 2021].<br>
*Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, Jianfeng Gao.*<br>
 [[PDF](https://arxiv.org/abs/2107.00641)] [[Github](https://github.com/microsoft/Focal-Transformer)]
 
**Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet.** [30th Nov. 2021] [ICCV, 2021].<br>
*Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan.*<br>
 [[PDF](https://arxiv.org/abs/2101.11986)] [[Github](https://github.com/yitu-opensource/T2T-ViT)]
 
**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions.** [11th Aug. 2021] [ICCV, 2021].<br>
*Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.*<br>
 [[PDF](https://arxiv.org/abs/2102.12122)] [[Github](https://github.com/whai362/PVT)]
 
 **Pvtv2: Improved baselines with pyramid vision transformer.** [9th Feb. 2022].<br>
*Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.*<br>
 [[PDF](https://arxiv.org/abs/2106.13797)] [[Github](https://github.com/whai362/PVT)]
 
**Rethinking Spatial Dimensions of Vision Transformers.** [18th Aug. 2021] [ICCV, 2021].<br>
*Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, Seong Joon Oh.*<br>
 [[PDF](https://arxiv.org/abs/2103.16302)] [[Github](https://github.com/naver-ai/pit)]
 
 **CvT: Introducing Convolutions to Vision Transformers.** [29th Mar. 2021] [ICCV, 2021].<br>
*Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2103.15808)] [[Github](https://github.com/microsoft/CvT)]
 
**Going deeper with Image Transformers.** [7th Apr. 2021] [ICCV, 2021].<br>
*Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou.*<br>
 [[PDF](https://arxiv.org/abs/2103.17239)] [[Github](https://github.com/facebookresearch/deit)]
 
**DeepViT: Towards Deeper Vision Transformer.** [19th Apr. 2021].<br>
*Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, Jiashi Feng.*<br>
 [[PDF](https://arxiv.org/abs/2103.11886)] [[Github](https://github.com/zhoudaquan/dvit_repo)]
 
**Refiner: Refining Self-attention for Vision Transformers.** [7th Jun. 2021].<br>
*Daquan Zhou, Yujun Shi, Bingyi Kang, Weihao Yu, Zihang Jiang, Yuan Li, Xiaojie Jin, Qibin Hou, Jiashi Feng.*<br>
 [[PDF](https://arxiv.org/abs/2106.03714)] [[Github](https://github.com/zhoudaquan/Refiner_ViT)]
 
**Vision Transformers with Patch Diversification.** [26th Apr. 2021].<br>
*Chengyue Gong, Dilin Wang, Meng Li, Vikas Chandra, Qiang Liu.*<br>
 [[PDF](https://arxiv.org/abs/2104.12753)] [[Github](https://github.com/ChengyueGongR/PatchVisionTransformer)]
 
**Generative Pretraining from Pixels.** [14th Nov. 2020] [ICML, 2020].<br>
*Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, Ilya Sutskever.*<br>
 [[PDF](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)] [[Github](https://github.com/openai/image-gpt)]
 
**MST: Masked Self-Supervised Transformer for Visual Representation.** [24th Oct. 2021] [NeurIPS, 2021].<br>
*Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang.*<br>
 [[PDF](https://arxiv.org/abs/2106.05656)]
 
**BEiT: BERT Pre-Training of Image Transformers.** [15th Jun. 2021] [ICLR, 2021].<br>
*Hangbo Bao, Li Dong, Furu Wei ·  Edit social preview.*<br>
 [[PDF](https://arxiv.org/abs/2106.08254)] [[Github](https://github.com/microsoft/unilm/tree/master/beit)]
 
**Masked Autoencoders Are Scalable Vision Learners.** [11th Nov. 2021].<br>
*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.*<br>
 [[PDF](https://arxiv.org/abs/2111.06377)] [[Github](https://github.com/facebookresearch/mae)]
 
**An Empirical Study of Training Self-Supervised Vision Transformers.** [16th Aug. 2021] [ICCV, 2021].<br>
*Xinlei Chen, Saining Xie, Kaiming He.*<br>
 [[PDF](https://arxiv.org/abs/2104.02057)] [[Github](https://github.com/facebookresearch/moco-v3)]
 
**Emerging Properties in Self-Supervised Vision Transformers.** [24th May. 2021] [ICCV, 2021].<br>
*Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin.*<br>
 [[PDF](https://arxiv.org/abs/2104.14294)] [[Github](https://github.com/facebookresearch/dino)]
 
**Self-Supervised Learning with Swin Transformers.** [10th May. 2021].<br>
*Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, Han Hu.*<br>
 [[PDF](https://arxiv.org/abs/2105.04553)] [[Github](https://github.com/SwinTransformer/Transformer-SSL)]
 

# Transformer for Detection
 
**End-to-End Object Detection with Transformers.** [18th May. 2020] [ECCV, 2020].<br>
*Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.*<br>
 [[PDF](https://arxiv.org/abs/2005.12872)] [[Github](https://github.com/facebookresearch/detr)]
 
**Pix2seq: A Language Modeling Framework for Object Detection.** [27th Mar. 2022] [ICLR, 2022].<br>
*Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, Geoffrey Hinton.*<br>
 [[PDF](https://arxiv.org/abs/2109.10852)] [[Github](https://github.com/google-research/pix2seq)]
 
**Deformable DETR: Deformable Transformers for End-to-End Object Detection.** [18th Mar. 2021] [ICLR, 2021].<br>
*Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.*<br>
 [[PDF](https://arxiv.org/abs/2010.04159)] [[Github](https://github.com/fundamentalvision/Deformable-DETR)]
 
**End-to-End Object Detection with Adaptive Clustering Transformer.** [18th Oct. 2021] [BMVC, 2021].<br>
*Minghang Zheng, Peng Gao, Renrui Zhang, Kunchang Li, Xiaogang Wang, Hongsheng Li, Hao Dong.*<br>
 [[PDF](https://arxiv.org/abs/2011.09315)] [[Github](https://github.com/gaopengcuhk/SMCA-DETR)]
 
**Pnp-detr: towards efficient visual analysis with transformers.** [2nd Mar. 2022] [ICCV, 2021].<br>
*Tao Wang, Li Yuan, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.*<br>
 [[PDF](https://arxiv.org/abs/2109.07036)] [[Github](https://github.com/twangnh/pnp-detr)]
 
**Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity.** [4th Mar. 2022] [ICLR, 2022].<br>
*Byungseok Roh, Jaewoong Shin, Wuhyun Shin, Saehoon Kim.*<br>
 [[PDF](https://arxiv.org/abs/2111.14330)] [[Github](https://github.com/kakaobrain/sparse-detr)]
 
**Fast Convergence of DETR with Spatially Modulated Co-Attention.** [19th Jan. 2021] [ICCV, 2021].<br>
*Peng Gao, Minghang Zheng, Xiaogang Wang, Jifeng Dai, Hongsheng Li.*<br>
 [[PDF](https://arxiv.org/abs/2101.07448)] [[Github](https://github.com/gaopengcuhk/SMCA-DETR)]
 
**Conditional DETR for Fast Training Convergence.** [19th Aug. 2021] [ICCV, 2021].<br>
*Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.*<br>
 [[PDF](https://arxiv.org/abs/2108.06152)] [[Github](https://github.com/atten4vis/conditionaldetr)]
 
**Anchor DETR: Query Design for Transformer-Based Object Detection.** [4th Jan. 2022].<br>
*Yingming Wang, Xiangyu Zhang, Tong Yang, Jian Sun.*<br>
 [[PDF](https://arxiv.org/abs/2109.07107)] [[Github](https://github.com/megvii-research/AnchorDETR)]
 
**Efficient DETR: Improving End-to-End Object Detector with Dense Prior.** [3th Apr. 2021].<br>
*Zhuyu Yao, Jiangbo Ai, Boxun Li, Chi Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2104.01318)]
 
**Dynamic detr: End-to-end object detection with dynamic attention.** [ICCV, 2021].<br>
*Xiyang Dai, Yinpeng Chen, Jianwei Yang, Pengchuan Zhang, Lu Yuan, Lei Zhang .*<br>
 [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.html)]
 
**Rethinking Transformer-based Set Prediction for Object Detection.** [12th Oct. 2021] [ICCV, 2021].<br>
*Zhiqing Sun, Shengcao Cao, Yiming Yang, Kris Kitani.*<br>
 [[PDF](https://arxiv.org/abs/2011.10881)] [[Github](https://github.com/edward-sun/tsp-detection)]
 
**You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection.** [27th Oct. 2021] [NeurIPS, 2021].<br>
*Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu.*<br>
 [[PDF](https://arxiv.org/abs/2106.00666)]  [[Github](https://github.com/hustvl/YOLOS)]
 
**UP-DETR: Unsupervised Pre-training for Object Detection with Transformers.** [7th Apr. 2021] [CVPR, 2021].<br>
*Zhigang Dai, Bolun Cai, Yugeng Lin, Junying Chen.*<br>
 [[PDF](https://arxiv.org/abs/2011.09094)]  [[Github](https://github.com/dddzg/up-detr)]
 
**FP-DETR: Detection Transformer Advanced by Fully Pre-training.** [29th Sep. 2021] [ICLR, 2021].<br>
*Wen Wang, Yang Cao, Jing Zhang, DaCheng Tao.*<br>
 [[PDF](https://openreview.net/pdf?id=yjMQuLLcGWK)]  
 
**DN-DETR: Accelerate DETR Training by Introducing Query DeNoising.** [2nd Mar. 2022] [CVPR, 2022].<br>
*Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M. Ni, Lei Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2203.01305)]  [[Github](https://github.com/IDEA-opensource/DN-DETR)]
 
**DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.** [7th Mar. 2022].<br>
*Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum.*<br>
 [[PDF](https://arxiv.org/abs/2203.03605)]  [[Github](https://github.com/IDEACVR/DINO)]
 
**Feature Pyramid Transformer.** [18th Jul. 2020] [ECCV, 2020].<br>
*Dong Zhang, Hanwang Zhang, Jinhui Tang, Meng Wang, Xiansheng Hua, Qianru Sun.*<br>
 [[PDF](https://arxiv.org/abs/2007.09451)]  [[Github](https://github.com/dongzhang89/FPT)]
 
**HRFormer: High-Resolution Vision Transformer for Dense Predict.** [7th Nov. 2021] [NeurIPS, 2021].<br>
*Yuhui Yuan, Rao Fu, Lang Huang, WeiHong Lin, Chao Zhang, Xilin Chen, Jingdong Wang.*<br>
 [[PDF](https://proceedings.neurips.cc//paper/2021/file/3bbfdde8842a5c44a0323518eec97cbe-Paper.pdf)]  [[Github](https://github.com/HRNet/HRFormer)]
 
**Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation.** [23rd Nov. 2021].<br>
*Jiaqi Gu, Hyoukjun Kwon, Dilin Wang, Wei Ye, Meng Li, Yu-Hsin Chen, Liangzhen Lai, Vikas Chandra, David Z. Pan.*<br>
 [[PDF](https://arxiv.org/abs/2111.01236)]
 

 
# Transformer for Segmentation
**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers.** [25th Jul. 2021] [CVPR 2021].<br>
*Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H. S. Torr, Li Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2012.15840)] [[Github](https://github.com/fudan-zvg/SETR)]
 
**TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.** [8th Feb. 2021].<br>
*Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou.*<br>
 [[PDF](https://arxiv.org/abs/2102.04306)] [[Github](https://github.com/Beckschen/TransUNet)]
 
**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** [28th Oct. 2021] [NeurIPS 2021].<br>
*Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.*<br>
 [[PDF](https://arxiv.org/abs/2105.15203)] [[Github](https://github.com/NVlabs/SegFormer)]
 
**Attention-Based Transformers for Instance Segmentation of Cells in Microstructures.** [20th Nov. 2020] [IEEE BIBM 2020].<br>
*Tim Prangemeier, Christoph Reich, Heinz Koeppl.*<br>
 [[PDF](https://arxiv.org/abs/2011.09763)]
 
**End-to-End Video Instance Segmentation with Transformers.** [8th Oct. 2021] [CVPR 2021].<br>
*Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, Huaxia Xia.*<br>
 [[PDF](https://arxiv.org/abs/2011.14503)] [[Github](https://github.com/Epiphqny/VisTR)]
 
**Instances as Queries.** [23rd May 2021] [ICCV 2021].<br>
*Yuxin Fang, Shusheng Yang, Xinggang Wang, Yu Li, Chen Fang, Ying Shan, Bin Feng, Wenyu Liu.*<br>
 [[PDF](https://arxiv.org/abs/2105.01928)] [[Github](https://github.com/hustvl/QueryInst)]
 
**ISTR: End-to-End Instance Segmentation with Transformers.** [3rd May 2021].<br>
*Jie Hu, Liujuan Cao, Yao Lu, Shengchuan Zhang, Yan Wang, Ke Li, Feiyue Huang, Ling Shao, Rongrong Ji.*<br>
 [[PDF](https://arxiv.org/abs/2105.00637)] [[Github](https://github.com/hujiecpp/ISTR)]
 
**SOLQ: Segmenting Objects by Learning Queries.** [30th Sep 2021] [NeurIPS 2021].<br>
*Bin Dong, Fangao Zeng, Tiancai Wang, Xiangyu Zhang, Yichen Wei.*<br>
 [[PDF](https://arxiv.org/abs/2106.02351)] [[Github](https://github.com/megvii-research/SOLQ)]
 
**MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers.** [12th Jul. 2021] [CVPR 2021].<br>
*Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.*<br>
 [[PDF](https://arxiv.org/abs/2012.00759)] [[Github](https://github.com/google-research/deeplab2)]
 
**Segmenter: Transformer for Semantic Segmentation.** [2nd Sep. 2021] [ICCV 2021].<br>
*Robin Strudel, Ricardo Garcia, Ivan Laptev, Cordelia Schmid.*<br>
 [[PDF](https://arxiv.org/abs/2105.05633)] [[Github](https://github.com/rstrudel/segmenter)]
 
**Per-Pixel Classification is Not All You Need for Semantic Segmentation.** [31st Oct. 2021] [NeurIPS 2021].<br>
*Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.*<br>
 [[PDF](https://arxiv.org/abs/2107.06278)] [[Github](https://github.com/facebookresearch/MaskFormer)]
 
# Transformer for 3D Visual Recognition

**Point Transformer.** [16th Dec. 2020] [ICCV 2021].<br>
*Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun.*<br>
 [[PDF](https://arxiv.org/abs/2012.09164)] [[Github](https://github.com/lucidrains/point-transformer-pytorch)]
 
**PCT: Point cloud transformer.** [17th Dec. 2020] [CVM 2021].<br>
*Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu.*<br>
 [[PDF](https://arxiv.org/abs/2012.09688)] [[Github](https://github.com/MenghaoGuo/PCT)]
 
**3DCTN: 3D Convolution-Transformer Network for Point Cloud Classification.** [2nd Mar. 2022].<br>
*Dening Lu, Qian Xie, Linlin Xu, Jonathan Li.*<br>
 [[PDF](https://arxiv.org/abs/2203.00828)]
 
**Fast Point Transformer.** [9th Dec. 2021] [CVPR 2022].<br>
*Chunghyun Park, Yoonwoo Jeong, Minsu Cho, Jaesik Park.*<br>
 [[PDF](https://arxiv.org/abs/2112.04702)]
 
**3D Object Detection with Pointformer.** [21th Dec. 2020] [CVPR 2021].<br>
*Xuran Pan, Zhuofan Xia, Shiji Song, Li Erran Li, Gao Huang.*<br>
 [[PDF](https://arxiv.org/abs/2012.11409)] [[Github](https://github.com/Vladimir2506/Pointformer)]
 
**Embracing Single Stride 3D Object Detector with Sparse Transformer.** [13th Dec. 2021].<br>
*Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, Zhaoxiang Zhang .*<br>
 [[PDF](https://arxiv.org/abs/2112.06375)] [[Github](https://github.com/tusimple/sst)]
 
**Embracing Single Stride 3D Object Detector with Sparse Transformer.** [13th Dec. 2021].<br>
*Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, Zhaoxiang Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2112.06375)] [[Github](https://github.com/tusimple/sst)]
 
**Voxel Transformer for 3D Object Detection.** [13th Sep. 2021] [ICCV 2021].<br>
*Jiageng Mao, Yujing Xue, Minzhe Niu, Haoyue Bai, Jiashi Feng, Xiaodan Liang, Hang Xu, Chunjing Xu.*<br>
 [[PDF](https://arxiv.org/abs/2109.02497)] [[Github](https://github.com/PointsCoder/VOTR)]
 
**Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds.** [19th Mar. 2022].<br>
*Chenhang He, Ruihuang Li, Shuai Li, Lei Zhang.*<br>
 [[PDF](https://arxiv.org/abs/2203.10314)] [[Github](https://github.com/skyhehe123/voxset)]
 
**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling.** [29th Nov. 2021] [CVPR 2022].<br>
*Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie zhou, Jiwen Lu.*<br>
 [[PDF](https://arxiv.org/abs/2111.14819)] [[Github](https://github.com/lulutang0608/Point-BERT)]
 
**Masked Autoencoders for Point Cloud Self-supervised Learning.** [13th Mar. 2022] [CVPR 2022].<br>
*Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie zhou, Jiwen Lu.*<br>
 [[PDF](https://arxiv.org/abs/2203.06604)] [[Github](https://github.com/Pang-Yatian/Point-MAE)]
 
**Masked Discrimination for Self-Supervised Learning on Point Clouds.** [21st Mar. 2022].<br>
*Haotian Liu, Mu Cai, Yong Jae Lee.*<br>
 [[PDF](https://arxiv.org/abs/2203.11183)] [[Github](https://github.com/haotian-liu/maskpoint)]
 
**An End-to-End Transformer Model for 3D Object Detection.** [16th Sep. 2021] [ICCV 2021].<br>
*Ishan Misra, Rohit Girdhar, Armand Joulin.*<br>
 [[PDF](https://arxiv.org/abs/2109.08141)] [[Github](https://github.com/facebookresearch/3detr)]
 
**Group-Free 3D Object Detection via Transformers.** [23rd Apr. 2021] [ICCV 2021].<br>
*Ze Liu, Zheng Zhang, Yue Cao, Han Hu, Xin Tong.*<br>
 [[PDF](https://arxiv.org/abs/2104.00678)] [[Github](https://github.com/zeliu98/Group-Free-3D)]
 
**Improving 3D Object Detection with Channel-wise Transformer.** [23rd Aug. 2021] [ICCV 2021].<br>
*Hualian Sheng, Sijia Cai, YuAn Liu, Bing Deng, Jianqiang Huang, Xian-Sheng Hua, Min-Jian Zhao.*<br>
 [[PDF](https://arxiv.org/abs/2108.10723)] [[Github](https://github.com/hlsheng1/ct3d)]
 
**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer.** [21st Mar. 2022] [CVPR 2022].<br>
*Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, Winston H. Hsu.*<br>
 [[PDF](https://arxiv.org/abs/2203.10981)] [[Github](https://github.com/kuanchihhuang/monodtr)]
 
**MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection.** [28th Mar. 2022] [CVPR 2022].<br>
*Renrui Zhang, Han Qiu, Tai Wang, Xuanzhuo Xu, Ziyu Guo, Yu Qiao, Peng Gao, Hongsheng Li.*<br>
 [[PDF](https://arxiv.org/abs/2203.13310)] [[Github](https://github.com/zrrskywalker/monodetr)]
 
**DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.** [13th Oct. 2021] [CRL 2022].<br>
*Yue Wang, Vitor Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, Justin Solomon.*<br>
 [[PDF](https://arxiv.org/abs/2110.06922)] [[Github](https://github.com/wangyueft/detr3d)]
 
**TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers.** [22nd Mar. 2022] [CVPR 2022].<br>
*Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, Chiew-Lan Tai.*<br>
 [[PDF](https://arxiv.org/abs/2203.11496)] [[Github](https://github.com/xuyangbai/transfusion)]
 
**PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers.** [19th Aug. 2021] [ICCV 2021].<br>
*Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, Jie zhou.*<br>
 [[PDF](https://arxiv.org/abs/2108.08839)] [[Github](https://github.com/yuxumin/PoinTr)]
 
**SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer.** [27th Oct. 2021] [ICCV 2021].<br>
*Peng Xiang, Xin Wen, Yu-Shen Liu, Yan-Pei Cao, Pengfei Wan, Wen Zheng, Zhizhong Han.*<br>
 [[PDF](https://arxiv.org/abs/2108.04444)] [[Github](https://github.com/allenxiangx/snowflakenet)]
 
**Deep Point Cloud Reconstruction.** [23rd Nov. 2021] [ICLR 2022].<br>
*Jaesung Choe, Byeongin Joung, Francois Rameau, Jaesik Park, In So Kweon.*<br>
 [[PDF](https://arxiv.org/abs/2111.11704)] [[Github](https://github.com/allenxiangx/snowflakenet)]

# Transformer for Multi-Sensory Data Stream
Coming soon... 

# Theoretical
Coming soon... 

# Citation

If you find the listing and survey useful for your work, please cite the paper:

```
@article{liu2021survey,
  title={A Survey of Visual Transformers},
  author={Liu, Yang and Zhang, Yao and Wang, Yixin and Hou, Feng and Yuan, Jin and Tian, Jiang and Zhang, Yang and Shi, Zhongchao and Fan, Jianping and He, Zhiqiang},
  journal={arXiv preprint arXiv:2111.06091},
  year={2021}
}

```

