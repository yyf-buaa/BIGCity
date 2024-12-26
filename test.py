from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import torch

# 加载GPT-2的预训练
tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
model = GPT2Model.from_pretrained("./models/gpt2")

# 定义并添加特殊token
special_tokens = {'additional_special_tokens': ['[CLS]', '[REG]']}
tokenizer.add_special_tokens(special_tokens)


# 调整模型嵌入层以适应新的特殊token
print(len(tokenizer), tokenizer)
model.resize_token_embeddings(len(tokenizer))

# 示例句子，包含自定义的特殊token
sentence = "predict the road segment on [CLS] based on supplement and input and [REG] result."

# 对句子进行编码（获得token IDs）
input_ids = tokenizer.encode(sentence, return_tensors='pt') # 返回PyTorch张量
print(input_ids, input_ids.shape)

print(model.wte(input_ids).shape, type(model.wte))

# 通过模型获得嵌入结果
with torch.no_grad():
    outputs = model(input_ids)

print(outputs.keys())
# 获取嵌入表示（最后一层的输出）
last_hidden_states = outputs.last_hidden_state

# 打印嵌入结果的形状
print("Shape of the embeddings:", last_hidden_states.shape)

# last_hidden_states 是一个 (batch_size, seq_length, hidden_size) 的张量
# 比如 (1, seq_length, 768)，其中768是GPT-2的hidden size
