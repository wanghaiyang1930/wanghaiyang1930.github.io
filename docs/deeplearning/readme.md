  📖 推荐阅读（如果想深入）

  1. 必读经典：
    - Smith (2015): "Cyclical Learning Rates"
    - Goyal et al. (2017): "Accurate Large Minibatch SGD"
  2. 工程实践：
    - He et al. (2019): "Bag of Tricks for Image Classification"
    - 包含大量实验验证的超参数选择技巧
  3. 理论深入：
    - Li et al. (2018): "Visualizing Loss Landscape"
    - 理解为什么不同阶段需要不同lr

  需要我提供这些论文的链接或者更详细解释某个部分吗？

1. 必读经典

Smith (2015): "Cyclical Learning Rates for Training Neural Networks"
  - 📄 arXiv: https://arxiv.org/abs/1506.01186
  - 🔗 PDF: https://arxiv.org/pdf/1506.01186.pdf
  - 💡 核心贡献：提出周期性学习率（CLR）和学习率范围测试（LR Range Test）

Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
  - 📄 arXiv: https://arxiv.org/abs/1706.02677
  - 🔗 PDF: https://arxiv.org/pdf/1706.02677.pdf
  - 💡 核心贡献：Linear Scaling Rule（学习率与batch size成正比）+ Warmup策略

---

2. 工程实践

He et al. (2019): "Bag of Tricks for Image Classification with Convolutional Neural Networks"
  - 📄 arXiv: https://arxiv.org/abs/1812.01187
  - 🔗 PDF: https://arxiv.org/pdf/1812.01187.pdf
  - 🔗 官方代码: https://github.com/dmlc/gluon-cv
  - 💡 核心内容：
    - Learning rate warmup
    - Cosine learning rate decay
    - Label smoothing
    - Mixup augmentation
    - 等大量实用技巧

---

3. 理论深入

Li et al. (2018): "Visualizing the Loss Landscape of Neural Networks"
  - 📄 arXiv: https://arxiv.org/abs/1712.09913
  - 🔗 PDF: https://arxiv.org/pdf/1712.09913.pdf
  - 🔗 官方代码: https://github.com/tomgoldstein/loss-landscape
  - 💡 核心贡献：3D可视化loss landscape，解释为什么不同阶段需要不同学习率
---
4. 额外推荐（与你的场景更相关）

Loshchilov & Hutter (2017): "Decoupled Weight Decay Regularization" (AdamW论文)
  - 📄 arXiv: https://arxiv.org/abs/1711.05101
  - 🔗 PDF: https://arxiv.org/pdf/1711.05101.pdf
  - 💡 你用的优化器，解释为什么AdamW比Adam更好

Yosinski et al. (2014): "How transferable are features in deep neural networks?"
  - 📄 arXiv: https://arxiv.org/abs/1411.1792
  - 🔗 PDF: https://arxiv.org/pdf/1411.1792.pdf
  - 💡 解释为什么backbone用0.1的lr_mult（不同层需要不同学习率）