<div align="center">

# M3-JEPA: Multimodal Alignment via Multi-gate MoE based on the Joint-Embedding Predictive Architecture

*Hongyang Lei<sup>1</sup>, Xiaolong Cheng<sup>1</sup>, Qi Qin<sup>2</sup>, Dan Wang<sup>1</sup>, Huazhen Huang<sup>3</sup>, Yetao Wu<sup>1</sup>, Qingqing Gu<sup>1</sup>, Luo Ji<sup>1</sup>*

<sup>1</sup>Geely AI Lab, Zhejiang, China  
<sup>2</sup>Peking University, Beijing, China  
<sup>3</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China

</div>

<p align="center">

  <a href="https://icml.cc/virtual/2025/poster/43776">
    <img src="https://img.shields.io/badge/ICML-Paper-brightgreen" alt="ICML">
  </a>
  <a href="https://arxiv.org/pdf/2409.05929">
    <img src="https://img.shields.io/badge/arXiv-2409.05929-brightgreen" alt="arXiv">
  </a>
    <a href="https://openreview.net/forum?id=tYwKQMMjJA">
  <img src="https://img.shields.io/badge/OpenReview-Forum-brightgreen" alt="OpenReview">
  </a>
  <a href="https://iclr.cc/virtual/2025/10000619">
    <img src="https://img.shields.io/badge/ICLR-Workshop%20Oral-brightgreen" alt="ICLR Workshop">
  </a>

</p>

📢 **Published in**:  
**Proceedings of the Forty-second International Conference on Machine Learning (ICML), 2025**

M3-JEPA is a novel framework for multimodal learning that addresses the limitations of token-space alignment, such as modality collapse and poor generalization. Instead of aligning modalities directly in the original token space, **M3-JEPA leverages a Joint Embedding Predictive Architecture (JEPA) to perform alignment in the latent embedding space.** It introduces a **Multi-Gate Mixture-of-Experts (MMoE) predictor** that adaptively disentangles and fuses modality-specific and shared features. This design not only improves alignment and representation quality, but also enhances transferability to unseen modalities and domains, making M3-JEPA a strong foundation for open-world self-supervised multimodal learning.

## 🔍 Overview of M3-Jepa

<p align="center">
  <img src="image/figure_1.jpg" alt="M3-Jepa Architecture" width="500"/>
</p>
The paradigm of M3-JEPA on any-to-any multi-modality tasks. The self-supervised learning is conducted with two encoding branches of input and output signals, as well as an MoE predictor which projects the input embedding into the output latent space. M3-JEPA is an energy-based model that minimizes both contrastive and regularization losses. M3-JEPA is also conditioned on the inherent information content (g) which maximizes the mutual information and minimizes the conditional entropy.

<p align="center">
  <img src="image/figure_2.jpg" alt="M3-Jepa Architecture" width="900"/>
</p>
Architecture of M3-Jepa: input and output are encoded by modality encoders and aligned on the latent space. A connector consisting of a multi-directional MoE is employed to project the input latent vector to the output space. The optimization is alternated between different uni-direction tasks step by step, and both contrastive learning (CL) and prediction learning (Pred) are implemented by loss components. The text-vision tasks are depicted as an experiment in the figure.

## 🚀 Highlights

### 🔍 Contributions of M3-JEPA

- ✅ **Any-to-Any Multimodal Alignment:** We propose a novel *any-to-any* multimodal alignment paradigm based on **JEPA**, mitigating modality collapse by aligning in the **latent embedding space** rather than the token space.

- ⚡ **Efficient MoE Predictor:** We introduce a computationally efficient **multi-gate Mixture-of-Experts (MoE)** architecture as the cross-modal predictor in JEPA, while **freezing most modality encoder parameters** to reduce training overhead.

- 🔄 **Disentangled Gating Mechanism:** Our design **disentangles the gating function** into **modality-specific** and **shared components**, and is supported by an **information-theoretic analysis** of its optimality.

- 🔁 **Alternating Task Optimization:** We optimize M3-JEPA using **alternating gradient descent (AGD)** over multiple **multi-directional multimodal tasks**, and provide a discussion of its **convergence behavior**.

- 🧪 **Extensive Evaluation:** Our experiments show **strong alignment accuracy and computational efficiency**, covering a wide range of modalities including **text, image, audio**, and more.

## 📄 Paper

**M3-Jepa: Multimodal Alignment via Multi-directional MoE based on the JEPA framework**  
👨‍💻 *Hongyang Lei, Xiaolong Cheng, Qi Qin, Dan Wang, Huazhen Huang, Yetao Wu, Qingqing Gu, Luo Ji*  
📍 Accepted at **ICML 2025 (Forty-second International Conference on Machine Learning)**

- 📄 [arXiv (2409.05929)](https://arxiv.org/pdf/2409.05929)  
- 📝 OpenReview: *coming soon*  
- 🔗 ICML Proceedings: *coming soon*

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/HongyangLL/M3-JEPA.git
cd m3-jepa

# Install dependencies
pip install -r requirements.txt
```

## 📢 Optimization & Release Status ##

We are currently optimizing M3-JEPA code for improved readability and maintainability in preparation for an upcoming release. Follow our progress in Issues or Commits. Stay tuned for a polished release soon! 🧹

This project provides a distributed training pipeline for a M3-JEPA model using PyTorch.

## 📌 TODO

- [ ] Release the initial version of M3-Jepa
- [x] Add arXiv citation and ICML acceptance info
- [ ] Release official ICML OpenReview and Proceedings links
- [ ] Upload training scripts and pretrained checkpoints
- [ ] Provide inference demo notebook
- [ ] ...
## 📚 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{lei2025m3jepamultimodalalignmentmultigate,
      title={M3-JEPA: Multimodal Alignment via Multi-gate MoE based on the Joint-Embedding Predictive Architecture}, 
      author={Hongyang Lei and Xiaolong Cheng and Qi Qin and Dan Wang and Kun Fan and Huazhen Huang and Qingqing Gu and Yetao Wu and Zhonglin Jiang and Yong Chen and Luo Ji},
      year={2025},
      eprint={2409.05929},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.05929}, 
}

