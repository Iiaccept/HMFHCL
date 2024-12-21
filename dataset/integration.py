import numpy as np




def calculate_CS(CSS, CGS):
    # 创建一个空矩阵来存储CS_ij的值，假设CSS和CGS具有相同的维度
    rows, cols = CSS.shape
    CS = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if CSS[i][j] != 0:
                CS[i][j] = (CSS[i][j] + CGS[i][j]) / 2
            else:
                CS[i][j] = CGS[i][j]

    return CS

CSS = np.loadtxt("drug_str_sim.txt")
CGS = np.loadtxt("GKGIP_drug.txt")


CS = calculate_CS(CSS, CGS)
np.savetxt(r'integration_drug.txt',  CS, delimiter='\t', fmt='%.9f')
print(np.array(CS))