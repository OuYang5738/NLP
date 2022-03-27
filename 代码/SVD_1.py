
import numpy as np
import matplotlib.pyplot as plt
'''
词的分布式表示：使用的是SVD
'''
M = np.array([[0,2,1,1,1,1,1,2,1,3],
              [2,0,1,1,1,0,0,1,1,2],
              [1,1,0,1,1,0,0,0,0,1],
              [1,1,1,0,1,0,0,0,0,1],
              [1,1,1,1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1,1,0,1],
              [1,0,0,0,0,1,0,1,0,1],
              [2,1,0,0,0,1,1,0,1,2],
              [1,1,0,0,0,0,0,1,0,1],
              [3,2,1,1,1,1,1,2,1,0]
              ])

def pmi(M , positive=True):
    col_t = M.sum(axis=0)
    row_t = M.sum(axis=1)
    t = col_t.sum()
    expected = np.outer(row_t, col_t) / t
    M = M/expected
    with np.errstate(divide='ignore'):
        M = np.log(M)
    M[np.isinf(M)] = 0.0
    if positive:
        M[M<0] = 0.0
    return  M


M_pmi = pmi(M)
U, s, Vh = np.linalg.svd(M_pmi)

words = ['我', '喜欢', '自然', '语言', '处理', '爱', '深度', '学习', '机器', '。']

for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])
# 常用正常中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# plt.savefig('1.jpg')

plt.show()
