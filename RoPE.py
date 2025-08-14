import math

THETA_BASE = 10000.0

"""
       10000 ** (-2(i - 1) / d)
"""

def get_rotary_frequencies(d_model):
    frequencies = []
    for j in range(d_model // 2):
        freq = 1.0 / (THETA_BASE ** (2 * j / d_model))
        frequencies.append(freq)
    return frequencies


def apply_rope(word_vector, position, d_model, frequencies):
    """
        对一个词向量应用旋转位置编码

        参数
        word_vector(list): 一个d_model维的词向量
        position 词语在句子中的绝对位置
        d_model 词向量的总维度
        frequencies 由 get_rotary_frequencies()函数生成的频率列表
    """

    rotated_vector = [0.0] * d_model
    """
        便利词向量中的每一对维度
    """
    for j in range(d_model // 2):
        v1 = word_vector[2 * j]
        v2 = word_vector[2 * j + 1]

        freq = frequencies[j]

        angle = position * freq

        rotated_vector[2 * j] = v1 * math.cos(angle) - v2 * math.sin(angle)
        rotated_vector[2 * j + 1] = v1 * math.sin(angle) - v2 * math.cos(angle)
    return rotated_vector


"""
    1, 旋转位置编码，解决了长距离依赖的问题
    2, 首先定了一个旋转频率，公式来自论文：10000 ** (-2(i - 1)/d)
    3, 计算夹角 angle = m * 公式(index)
    4, x1, x2，词向量的值，俩俩对应
    5, x' = x1 * cos(angle) - x2 * sin(angle)
    6, y' = x1 * sin(angle) - x2 * cos(angle)
    

"""


# 假设我们有一个句子，词向量维度为4
d_model = 4

# 预先计算好旋转频率表，这是静态的，只需要计算一次
frequencies = get_rotary_frequencies(d_model)

# 假设我们有三个词，每个词都有一个原始词向量
query_vector_0 = [0.1, 0.2, 0.3, 0.4] # 句子中的第一个词（早）
key_vector_0   = [0.1, 0.2, 0.3, 0.4] # 句子中的第一个词（早）

query_vector_1 = [0.5, 0.6, 0.7, 0.8] # 句子中的第二个词（上）
key_vector_1   = [0.5, 0.6, 0.7, 0.8] # 句子中的第二个词（上）

query_vector_2 = [0.9, 1.0, 1.1, 1.2] # 句子中的第三个词（好）
key_vector_2   = [0.9, 1.0, 1.1, 1.2] # 句子中的第三个词（好）

q0_rope = apply_rope(query_vector_0, position=0, d_model=d_model, frequencies=frequencies)
k0_rope = apply_rope(key_vector_0,   position=0, d_model=d_model, frequencies=frequencies)
print(f"位置0（早）的词向量，旋转后：\n  Q: {q0_rope}\n  K: {k0_rope}")

