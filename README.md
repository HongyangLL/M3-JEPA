# 🔍 M3-Jepa: Scalable Multimodal Alignment via Latent Space Prediction

**M3-Jepa** is a scalable multimodal alignment framework that moves beyond token-level matching by aligning representations in the **latent space**. Built upon the Joint-Embedding Predictive Architecture (JEPA), M3-Jepa introduces a **multi-directional Mixture-of-Experts (MoE)** predictor and optimizes alignment via alternating uni-directional tasks. This approach maximizes mutual information and effectively mitigates modality bias. Extensive experiments show that M3-Jepa achieves **state-of-the-art performance**, strong generalization across unseen modalities and domains, and high computational efficiency. M3-Jepa offers a promising path for **self-supervised multimodal learning** and **open-world understanding**.

## 🚀 Highlights

- ✅ We propose a novel modality-agnostic multi-modal alignment paradigm, with the alignment conducted on the latent space, which is computationally efficient especially when employed as a retriever.
- ✅ We leverage multi-directional MoE as the cross-modal connector, optimizing by alternating the gradient descent between different unidirectional alignment task.
- ✅ We derive an information-theoretical explanation analysis, demonstrating the optimality of M3-Jepa.
- ✅  Our experimental results demonstrate remarkable multi-modal alignment accuracy and efficiency, encompassing text, image and audio modalities.

## 📄 Paper

> **M3-Jepa: Scalable Multimodal Alignment via Latent Space Prediction**  
> Hongyang Lei, et al.  
> Accepted at *ICML 2025*.  
> 📄 [arXiv (2409.05929)](https://arxiv.org/pdf/2409.05929)  
> 📝 OpenReview: *coming soon*  
> 🔗 ICML proceedings link: *coming soon*

## 🛠 Installation

```bash
# Clone the repository
git clone [https://github.com/HongyangLL/M3-JEPA/]
cd m3-jepa

# Install dependencies
pip install -r requirements.txt

