# 🔍 M3-Jepa: Multimodal Alignment via Multi-directional MoE based on the JEPA framework

**M3-Jepa** is a scalable multimodal alignment framework that moves beyond token-level matching by aligning representations in the **latent space**. Built upon the Joint-Embedding Predictive Architecture (JEPA), M3-Jepa introduces a **multi-directional Mixture-of-Experts (MoE)** predictor and optimizes alignment via alternating uni-directional tasks. This approach maximizes mutual information and effectively mitigates modality bias. Extensive experiments show that M3-Jepa achieves **state-of-the-art performance**, strong generalization across unseen modalities and domains, and high computational efficiency. M3-Jepa offers a promising path for **self-supervised multimodal learning** and **open-world understanding**.

## 🚀 Highlights

- ✅ We propose a novel modality-agnostic multi-modal alignment paradigm, with the alignment conducted on the latent space, which is computationally efficient especially when employed as a retriever.
- ✅ We leverage multi-directional MoE as the cross-modal connector, optimizing by alternating the gradient descent between different unidirectional alignment task.
- ✅ We derive an information-theoretical explanation analysis, demonstrating the optimality of M3-Jepa.
- ✅  Our experimental results demonstrate remarkable multi-modal alignment accuracy and efficiency, encompassing text, image and audio modalities.

## 📄 Paper

**M3-Jepa: Multimodal Alignment via Multi-directional MoE based on the JEPA framework**  
👨‍💻 *Hongyang Lei, Xiaolong Cheng, Qi Qin, Dan Wang, Huazhen Huang, Yetao Wu, Qingqing Gu, Luo Ji*  
📍 Accepted at **ICML 2025 (Forty-second International Conference on Machine Learning)**

- 📄 [arXiv (2409.05929)](https://arxiv.org/pdf/2409.05929)  
- 📝 OpenReview: *coming soon*  
- 🔗 ICML Proceedings: *coming soon*

## 🔍 Overview of M3-Jepa
Paradigm of our M3-Jepa. The self-supervised learning is conducted with two encoding branches of input and output signals, and a multi-directional MoE predictor to match the input latent embedding into the target latent embedding by minimizing contrastive and prediction losses. The MoE predictor is conditioned on the multi-modal routed information which is learned with information entropy minimization. Training is performed with alternative any-to-any multi-modality tasks.
<p align="center">
  <img src="image/m3_jepa_method.png" alt="M3-Jepa Architecture" width="500"/>
</p>

Architecture of M3-Jepa: input and output are encoded by modality encoders and aligned on the latent space. A connector consisting of a multi-directional MoE is employed to project the input latent vector to the output space. The optimization is alternated between different uni-direction tasks step by step, and both contrastive learning (CL) and prediction learning (Pred) are implemented by loss components. The text-vision tasks are depicted as an experiment in the figure.
<p align="center">
  <img src="image/m3_jepa_method2.png" alt="M3-Jepa Architecture" width="800"/>
</p>

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/HongyangLL/M3-JEPA.git
cd m3-jepa

# Install dependencies
pip install -r requirements.txt
```

## 📚 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{lei2024alt,
  title={Alt-MoE: Multimodal Alignment via Alternating Optimization of Multi-directional MoE with Unimodal Models},
  author={Lei, Hongyang and Cheng, Xiaolong and Wang, Dan and Qin, Qi and Huang, Huazhen and Wu, Yetao and Gu, Qingqing and Jiang, Zhonglin and Chen, Yong and Ji, Luo},
  journal={arXiv preprint arXiv:2409.05929},
  year={2024}
}

