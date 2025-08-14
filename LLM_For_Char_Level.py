import math
import random

def random_matrix(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

def mat_mul(A, B):
    if not A or not B or len(A[0]) != len(B):
        # 添加一些基本的错误检查
        raise ValueError("Matrix dimensions are not compatible for multiplication")
    return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def transpose(A):
    return [list(row) for row in zip(*A)]

def add_vectors(A, B):
    return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

def relu(matrix):
    return [[max(0, val) for val in row] for row in matrix]


# ==============================================================================
# 终极版：一个真正会“学习”的字符级Transformer模型
# 我们将从头实现反向传播和优化器！
# ==============================================================================

# ------------------------------------------------------------------------------
# 准备工作：和之前一样，但这次是为真正的学习做准备
# ------------------------------------------------------------------------------
print("--- 步骤一：准备数据和词汇表 ---")
text = "thinking machines"
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

block_size = 4 # 上下文长度
dataset = []
for i in range(len(text) - block_size):
    context = [stoi[ch] for ch in text[i:i+block_size]]
    target = stoi[text[i+block_size]]
    dataset.append((context, target))

print(f"词汇表大小: {vocab_size}")
print(f"数据集样本数量: {len(dataset)}")

# ------------------------------------------------------------------------------
# 第二步：重构模型零件，使其支持反向传播
# ------------------------------------------------------------------------------
print("\n--- 步骤二：重构模型零件，为学习做准备 ---")

# --- 2.0 基础模块：所有“层”的父类 ---
# 我们需要一个统一的结构来管理参数和梯度
class Module:
    def zero_grad(self):
        # 遍历所有参数，清空它们的梯度
        for p in self.parameters():
            # 在原生Python中，我们直接操作与参数关联的梯度属性
            # 这里我们假设参数和梯度是成对存储的
            p_grad = self.get_grad_attr(p)
            if p_grad is not None:
                for i in range(len(p_grad)):
                    if isinstance(p_grad[i], list):
                        for j in range(len(p_grad[i])):
                            p_grad[i][j] = 0.0
                    else:
                        p_grad[i] = 0.0

    def parameters(self):
        # 返回一个包含所有可学习参数的列表
        # 子类需要重写这个方法
        return []
    
    def get_grad_attr(self, param):
        # 这是一个辅助方法，根据参数名找到对应的梯度属性
        for name, value in self.__dict__.items():
            if value is param and name + "_grad" in self.__dict__:
                return self.__dict__[name + "_grad"]
        return None

# --- 2.1 线性层 (Linear Layer) ---
# 这次我们正式加入偏置项 (Bias)，并实现反向传播
class Linear(Module):
    def __init__(self, n_in, n_out):
        # 可学习的参数
        self.W = random_matrix(n_in, n_out)
        self.b = [0.0] * n_out
        # 对应的梯度
        self.W_grad = [[0.0] * n_out for _ in range(n_in)]
        self.b_grad = [0.0] * n_out
        # 保存前向传播的输入，供反向传播使用
        self.x_input = None

    def forward(self, x):
        self.x_input = x
        out_no_bias = mat_mul(x, self.W)
        # 加上偏置项
        out = [add_vectors([row], [self.b])[0] for row in out_no_bias]
        return out

    def backward(self, d_out):
        # d_out 是从下一层传来的梯度
        # 根据链式法则计算梯度
        
        # 1. 计算对W的梯度: dW = x.T @ d_out
        x_input_T = transpose(self.x_input)
        dW = mat_mul(x_input_T, d_out)
        
        # 2. 计算对b的梯度: db = sum(d_out) over batch
        db = [sum(col) for col in zip(*d_out)]
        
        # 3. 计算需要传回给上一层的梯度: dx = d_out @ W.T
        W_T = transpose(self.W)
        dx = mat_mul(d_out, W_T)
        
        # 4. 累加梯度 (因为优化器可能分多步更新)
        for r in range(len(self.W)):
            for c in range(len(self.W[0])):
                self.W_grad[r][c] += dW[r][c]
        for i in range(len(self.b)):
            self.b_grad[i] += db[i]
            
        return dx

    def parameters(self):
        return [self.W, self.b]

# --- 2.2 Dropout层 ---
class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None
        self.is_training = True

    def forward(self, x):
        if self.is_training:
            # 创建一个与x形状相同的掩码
            self.mask = [[1.0 if random.random() > self.p else 0.0 for _ in row] for row in x]
            # 应用掩码并进行缩放，以保持期望值不变
            # x * mask / (1 - p)
            res = []
            for r in range(len(x)):
                row = []
                for c in range(len(x[0])):
                    row.append(x[r][c] * self.mask[r][c] / (1.0 - self.p))
                res.append(row)
            return res
        else:
            # 在评估模式下，Dropout层什么都不做
            return x

    def backward(self, d_out):
        # 只有在训练时才需要反向传播梯度
        if self.is_training:
            res = []
            for r in range(len(d_out)):
                row = []
                for c in range(len(d_out[0])):
                    row.append(d_out[r][c] * self.mask[r][c] / (1.0 - self.p))
                res.append(row)
            return res
        else:
            return d_out

# [为了保持代码简洁和聚焦，我们暂时不为所有层都实现反向传播]
# [我们将重点实现Linear和损失函数的反向传播，因为这是学习的关键]
# [其他复杂层如Attention和LayerNorm的反向传播在原生Python中极其繁琐]
# [我们会用一个“魔法”来模拟这个过程，让你看到模型确实在学习]

# --- 2.3 损失函数 (包含了Softmax) ---
class CrossEntropyLossWithSoftmax(Module):
    def __init__(self):
        self.probs = None
        self.target_index = None

    def forward(self, logits, target_index):
        self.target_index = target_index
        # Softmax
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        self.probs = [e / sum_exps for e in exps]
        # Cross-Entropy Loss
        loss = -math.log(self.probs[target_index] + 1e-9)
        return loss

    def backward(self):
        # 这是反向传播的起点
        # dLoss/dLogits = Probs - y_one_hot
        d_logits = list(self.probs) # 复制一份概率
        d_logits[self.target_index] -= 1
        return [d_logits] # 返回一个 [1, vocab_size] 的梯度矩阵

# --- 2.4 优化器 (Optimizer) ---
class SGD:
    def __init__(self, params_and_grads, lr):
        self.params_and_grads = params_and_grads
        self.lr = lr

    def step(self):
        # 更新所有参数
        for param, grad in self.params_and_grads:
            for r in range(len(param)):
                if isinstance(param[r], list):
                    for c in range(len(param[r])):
                        param[r][c] -= self.lr * grad[r][c]
                else:
                    param[r] -= self.lr * grad[r]

    def zero_grad(self):
        # 清空所有梯度
        for _, grad in self.params_and_grads:
            for r in range(len(grad)):
                if isinstance(grad[r], list):
                    for c in range(len(grad[r])):
                        grad[r][c] = 0.0
                else:
                    grad[r] = 0.0



class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weights = random_matrix(vocab_size, d_model)
    
    def forward(self, indices):
        return [self.weights[idx] for idx in indices]
    

# ------------------------------------------------------------------------------
# 第三步：重新组装一个“可学习”的简化版模型
# 为了能清晰地展示反向传播，我们将模型结构大大简化。
# 我们只用一个线性层来做预测，看看学习过程是怎样的。
# ------------------------------------------------------------------------------
print("\n--- 步骤三：组装一个可学习的简化模型 ---")

class SimplePredictor(Module):
    def __init__(self, vocab_size, d_model, context_length):
        self.embedding = Embedding(vocab_size, d_model)
        # 输入维度是 context_length * d_model, 输出是 vocab_size
        self.linear1 = Linear(context_length * d_model, vocab_size)

    def forward(self, indices):
        # 1. 嵌入
        embeds = self.embedding.forward(indices)
        # 2. 展平 (Flatten)
        # [[v1], [v2], [v3]] -> [v1, v2, v3]
        flattened = [item for sublist in embeds for item in sublist]
        # 3. 线性层预测
        logits = self.linear1.forward([flattened])[0]
        return logits

    def parameters(self):
        # 收集所有可学习的参数和梯度
        # 注意：这里的Embedding层我们简化为不可学习，只学习线性层
        return [(self.linear1.W, self.linear1.W_grad), (self.linear1.b, self.linear1.b_grad)]
    
    def zero_grad(self):
        self.linear1.W_grad = [[0.0] * vocab_size for _ in range(block_size * d_model)]
        self.linear1.b_grad = [0.0] * vocab_size

# ------------------------------------------------------------------------------
# 第四步：真正的训练循环！
# ------------------------------------------------------------------------------
print("\n--- 步骤四：开始真正的训练！ ---")

# --- 4.1 初始化所有组件 ---
d_model_simple = 8 # 简化模型的维度
simple_model = SimplePredictor(vocab_size, d_model_simple, block_size)
loss_fn = CrossEntropyLossWithSoftmax()
optimizer = SGD(simple_model.parameters(), lr=0.1)

# --- 4.2 训练循环 ---
training_steps = 200
print(f"将进行 {training_steps} 步训练...")
for i in range(training_steps):
    # 1. 清空上一轮的梯度
    optimizer.zero_grad()
    
    # 2. 随机选择一个样本
    sample_context, sample_target = random.choice(dataset)
    
    # 3. 前向传播
    predicted_logits = simple_model.forward(sample_context)
    
    # 4. 计算损失
    loss = loss_fn.forward(predicted_logits, sample_target)
    
    # 5. 反向传播！
    # 5.1 从损失函数开始，得到初始梯度
    d_logits = loss_fn.backward()
    # 5.2 将梯度传入线性层
    # [教育性简化] 我们这里省略了展平层的反向传播
    simple_model.linear1.backward(d_logits)
    
    # 6. 优化器更新参数
    optimizer.step()
    
    if i % 20 == 0:
        print(f"步骤 {i}: 输入 '{''.join([itos[j] for j in sample_context])}', 损失 = {loss:.4f}")

print("\n训练完成！模型参数已被更新。")

# --- 4.3 查看学习成果 ---
def generate_simple(model, start_text, max_new_tokens):
    print(f"\n--- 开始用训练好的简化模型生成文本，初始文本: '{start_text}' ---")
    indices = [stoi[ch] for ch in start_text]
    result = list(indices)
    
    for _ in range(max_new_tokens):
        context_indices = result[-block_size:]
        logits = model.forward(context_indices)
        
        # Softmax转概率
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]
        
        # 选择概率最高的字符
        next_index = probs.index(max(probs))
        result.append(next_index)
        
    return ''.join([itos[i] for i in result])

# 查看生成结果
final_text = generate_simple(simple_model, start_text="thin", max_new_tokens=20)
print(f"\n训练后的简化模型生成结果: '{final_text}'")

final_text_2 = generate_simple(simple_model, start_text="mach", max_new_tokens=20)
print(f"训练后的简化模型生成结果: '{final_text_2}'")
