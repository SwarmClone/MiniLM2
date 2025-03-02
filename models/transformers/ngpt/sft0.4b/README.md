# MiniLM2 nGPT 0.4b Instruct

这是一个基于NVIDIA的[nGPT](https://github.com/NVIDIA/ngpt)的小型语言模型，是[蜂群克隆计划](https://github.com/SwarmClone)的核心。此模型经过了[匠数科技SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files)的微调。

## 使用方法

参见[GitHub仓库](https://github.com/SwarmClone/MiniLM2/tree/huggingface)中`minilm2/utils/test_sft_model.py`。因为包含了模型代码，请注意设置`trust_remote_code=True`。

## 模型输出实例

系统提示词：`AI是一个名叫MiniLM2的小型语言模型。AI是人类的助手，会回答用户的问题并遵守用户的指令。`

输入：`请向我介绍什么是大语言模型。`

输出：
```
大语言模型是一种能够处理和分析自然语言的模型，它使用了一些神经网络技术，以便能够生成有意义的语言回答。大语言模型通常被用于语言模型的训练，例如机器翻译、语音识别和文本分类等任务。

大语言模型的主要优点是能够处理复杂的自然语言问题，并且可以处理非常长的自然语言文本。它可以学习到自然语言的语法、词汇和语义，并能够生成自然流畅的回答。大语言模型已经被广泛应用于机器翻译、语音识别和文本分类等任务中。
```
