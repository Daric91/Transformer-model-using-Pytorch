# Transformer-model-using-Pytorch
To make a transfomer language model using pytorch

Dataset of training: https://huggingface.co/datasets/Digilidaahz/chat_dataset_2B_cha/

# What is a transformer?
The Transformer is a model that uses self-attention to process all words in a sentence at the same time, 
so it understands context better and doesn’t need recurrent or convolutional layers.

## Structure of a transformer
**Input:**
* Tokenizer
* Embedding
* Position Embedding

**Transfomer block**
* LayerNorm
* Multihead attention
* +Residual
* LayerNorm
* Feed Forward Network
* +Residual

**Output**
* Linear layer
* Vocabulary size logits
* Softmax
* possibility every token

**Training...**

## 1. Tokenizer
Tokenizer is a tool that converts string characters into token ID's, for example:
a = 0
b = 1
c = 2
d = 3
(and so on...)

`code`

