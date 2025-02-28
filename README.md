# SwarmClone 从零开始·蜂群克隆计划
## MiniLM2
这是一个基于NVIDIA的[nGPT](https://arxiv.org/abs/2410.01131)的小型语言模型，
其前身为[KyvYang](https://github.com/kyv001)的个人项目[minilm](https://github.com/kyv001/minilm)。
但是为了性能而将重新实现的BPE换成了HuggingFace的[Tokenizes](https://github.com/huggingface/tokenizers)提供的BPE，
模型本身的实现并不变，而重新实现了训练循环和数据处理部分。
