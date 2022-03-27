import numpy as np
'''
词的分布式表示：使用点互信息 PMII
'''
M = np.array([[0,2,1,1,1],
              [2,0,1,1,1],
              [1,1,0,1,1],
              [1,1,1,0,1],
              [1,1,1,1,0]])

def pmi(M , positive=True):
    col_t = M.sum(axis=0)
    row_t = M.sum(axis=1)
    print('col\n',col_t)
    print('row\n',row_t)
    t = col_t.sum()
    print('outer\n',np.outer(row_t , col_t))
    expected = np.outer(row_t , col_t) / t
    M = M/expected
    with np.errstate(divide = 'ignore'):
        M = np.log(M)
        print('M\n',M)
    M[np.isinf(M)] = 0.0
    if positive:
        M[M<0] = 0.0
    return  M

M_pmi = pmi(M)
np.set_printoptions(precision=2)
print(M_pmi)