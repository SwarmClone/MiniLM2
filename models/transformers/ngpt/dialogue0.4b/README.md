# MiniLM2 nGPT 0.4b Dialogue

这是一个基于NVIDIA的[nGPT](https://github.com/NVIDIA/ngpt)的小型语言模型，是[蜂群克隆计划](https://github.com/SwarmClone)的核心。此模型在Instruct模型基础上经过了MagicData-CLAM的自然语言对话微调。

## 使用方法

参见[GitHub仓库](https://github.com/SwarmClone/MiniLM2/tree/huggingface)中`minilm2/utils/test_dialogue_model.py`。必须使用仓库中的代码才能正常推理模型，因为此模型架构并未被`Transformers`库支持。

## 模型输出实例

系统提示词：`AI是一个名叫知络的16岁女孩。AI与人类闲聊。`

输入：`你好！今天过得怎样？`

输出：`我今天过得挺好的，我们一起玩了好几天。`
