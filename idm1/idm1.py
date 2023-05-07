import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# 导入数据,并读取

data = pd.read_csv(r'D:\Ga\genchidui.csv')
sample_size = 80  # 随机选取80组跟驰对
selected_data = pd.DataFrame(columns=data.columns)  # 创建空DataFrame对象
while len(selected_data) < sample_size:
    index = random.randint(0, len(data) - 1)  # 随机选择一个索引
    row = data.iloc[index]  # 提取数据行
    selected_data = pd.concat([selected_data, row.to_frame().T], ignore_index=True)


p1 = selected_data.iloc[:, 0]
s1 = selected_data.iloc[:, 1]
p2 = selected_data.iloc[:, 2]
s2 = selected_data.iloc[:, 3]
a2 = selected_data.iloc[:, 4]
# p1 前车位置
# s1 前车速度
# p2 后车位置
# s2 后车速度
# a2 后车加速度


v = s1
v1 = s2
distance = p1-p2


# 定义跟驰模型
def idm_model(x,  a):
    # s: 车头间距
    # T: 最小安全车头时距
    # a: 最大加速度
    # b: 最大减速度
    # ve: 期望车速
    s = x(1) + x(2) * (v * (v - v1)) / (2 * (x(3) * x(4)) ** 0.5)
    a_sim = a * (1 - (v / x(5)) ** 4 - (s / distance) ** 2)
    return a_sim


lb = [1, 0.25, 1, 1, 10]  # 决策变量下界
ub = [20, 2.5, 5, 5, 25]  # 决策变量上界


# 定义Theil's函数适应度函数
def fitness_func(a_sim):
    z = np.sum(np.square(a2 - a_sim))
    real = np.sum(np.square(a2))
    sim = np.sum(np.square(a_sim))
    f = (z / 199) ** 0.5 / ((real / 199) ** 0.5 + (sim / 199) ** 0.5)
    return f


# 遗传算法参数
pop_size = 50  # 种群大小
chromo_len = 5  # 染色体长度
pc = 0.9  # 交叉概率
pm = 0.05  # 变异概率
max_gen = 500  # 最大迭代次数


# 初始化种群
def init_population():
    pop = []
    for i in range(pop_size):
        chromo = [random.uniform(0.0, 1.0) for _ in range(chromo_len)]
        pop.append(chromo)
    return pop


# 选择操作
def selection(pop, fitness):
    new_pop = []
    for i in range(pop_size):
        idx1 = random.randint(0, pop_size - 1)
        idx2 = random.randint(0, pop_size - 1)
        if fitness[idx1] < fitness[idx2]:
            new_pop.append(pop[idx1])
        else:
            new_pop.append(pop[idx2])
    return new_pop


# 交叉操作
def crossover(pop):
    new_pop = []
    for i in range(0, pop_size, 2):
        chromo1 = pop[i]
        chromo2 = pop[i+1]
        if random.random() < pc:
            point = random.randint(1, chromo_len - 1)
            new_chromo1 = chromo1[:point] + chromo2[point:]
            new_chromo2 = chromo2[:point] + chromo1[point:]
            new_pop.append(new_chromo1)
            new_pop.append(new_chromo2)
        else:
            new_pop.append(chromo1)
            new_pop.append(chromo2)
    return new_pop


# 变异操作
def mutation(pop):
    new_pop = []
    for chromo in pop:
        new_chromo = chromo.copy()
        for i in range(chromo_len):
            if random.random() < pm:
                new_chromo[i] = random.uniform(0.0, 1.0)
        new_pop.append(new_chromo)
    return new_pop


# 主函数
def genetic_algorithm(a_sim):
    pop = init_population()
    for gen in range(max_gen):
        fitness = fitness_func(a_sim)
        best_idx = np.argmin(fitness)
        print("Generation:", gen, " Best error:", fitness[best_idx], " Params:", pop[best_idx])
        if fitness[best_idx] < 1e-3:
            break
    return


plt.plot('fitness')
plt.xlabel('Generation')
plt.ylabel('fitness value')
plt.show()
