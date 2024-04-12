# %% [markdown]
# Подключаем библиотеки для работы с массивами и построения графиков.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from IPython.display import Markdown

# %% [markdown]
# Считываем матрицу замеров из файла "data.csv"

# %%
data_matrix = np.loadtxt("data_homework_Dementeva.csv", delimiter=",")
n,m=data_matrix.shape
print(f"Матрица замеров:\n"
      f"{pd.DataFrame(data=data_matrix)}")

# %% [markdown]
# Определим общее число испытаний $N$ (пар событий $x_i y_i$)

# %%
N=data_matrix.sum()
Markdown(rf"""
$N={N}$""")

# %% [markdown]
# $P(x_i,y_j)=\frac{n_{ij}}{N}$ - совместная вероятность события $x_i,y_j$. На основе этой формулы построим матрицу совместных вероятностей событий

# %%
joint_probabilities_matrix=data_matrix/N
print(f"Матрица совместных вероятностей:\n"
      f"{pd.DataFrame(data=joint_probabilities_matrix)}")

# %% [markdown]
# Рассчитаем вектора $P(x_i)$ и $P(y_i)$

# %%
p_x=np.zeros(n)
p_y=np.zeros(m)
for i in range(n):
    for j in range(m):
        p_x[i]+=joint_probabilities_matrix[i][j]
for j in range(m):
    for i in range(n):
        p_y[j]+=joint_probabilities_matrix[i][j]
Markdown(rf"""
$P(x_i)=${p_x}$\\$
$P(y_i)=${p_y}
""")


# %% [markdown]
# По формулам, указанным ниже найдём входную и выходную энтропию, энтропию сложного опыта XY и количество информации, которое несёт о событии X наблюдаемое событие Y.
# $\\H(X)=-\sum_{i=1}^{4}P(x_i)log_2 P(x_i)\\$
# $H(Y)=-\sum_{j=1}^{4}P(y_j)log_2 P(y_j)\\$
# $H(X,Y)=-\sum_{i=1}^{4}\sum_{j=1}^{4}P(x_i,y_j)log_2 P(x_i,y_j)\\$
# $I(X,Y)=H(X)+H(Y)-H(X,Y)\\$

# %%
#функция для вычисления логарифма
def log_2(x):
    if x==0:
        return 0
    else: 
        return np.log2(x)
    
H_X=0.0
H_Y=0.0
H_X_Y=0.0
for i in range(n):
    H_X-=p_x[i]*log_2(p_x[i])
for j in range(m):
    H_Y-=p_y[j]*log_2(p_y[j])
for i in range(n):
    for j in range(m):
        H_X_Y-=joint_probabilities_matrix[i][j]*log_2(joint_probabilities_matrix[i][j])
H_X=np.round(H_X,6)
H_Y=np.round(H_Y,6)
H_X_Y=np.round(H_X_Y,6)
I_X_Y=np.round(H_X+H_Y-H_X_Y,6)
Markdown(rf"""
$H(X)={H_X}\\$
$H(Y)={H_Y}\\$
$H(X,Y)={H_X_Y}\\$
$I(X,Y)=H(X)+H(Y)-H(X,Y)={H_X}+{H_Y}-{H_X_Y}={I_X_Y}$
""")

# %% [markdown]
# Найдём условную энтропию события X при условии Y и условную энтропию события Y при условии X.
# $\\H(X/Y)=H(X)-I(X,Y)\\$
# $H(Y/X)=H(Y)-I(X,Y)$

# %%
H_X_cond_Y=np.round(H_X-I_X_Y,6)
H_Y_cond_X=np.round(H_Y-I_X_Y,6)
Markdown(rf"""
$H(X/Y)={H_X}-{I_X_Y}={H_X_cond_Y}\\$
$H(Y/X)={H_Y}-{I_X_Y}={H_Y_cond_X}$
""")

# %% [markdown]
# Для проверки составим матрицу условных вероятностей.
# $\\P(x_i/y_j)=\frac{P(x_i,y_j)}{P(y_j)}$

# %%
conditional_probability_matrix=np.zeros((n,m))
for i in range(n):
    for j in range(m):
        if p_y[j]==0:
            conditional_probability_matrix[i][j]=0
        else:
            conditional_probability_matrix[i][j]=joint_probabilities_matrix[i][j]/p_y[j]
print(f"Матрица условных вероятностей:\n"
      f"{pd.DataFrame(data=conditional_probability_matrix)}")

# %% [markdown]
# Найдём условную энтропию события X при условии Y:
# $\\H(X/Y)=-\sum_{i=1}^{4}\sum_{j=1}^{4}P(x_i,y_j)log_2 P(x_i/y_j)\\$
# $I_п(X,Y)=H(X)-H(X/Y)$

# %%
H_X_cond_Y_check=0.0
I_X_Y_check=0.0
for i in range(n):
    for j in range(m):
        H_X_cond_Y_check-=joint_probabilities_matrix[i][j]*log_2(conditional_probability_matrix[i][j])
H_X_cond_Y_check=np.round(H_X_cond_Y_check,6)
I_X_Y_check=np.round(H_X-H_X_cond_Y_check,6)
Markdown(rf"""
$H_п(X/Y)={H_X_cond_Y_check}\\$
$I_п(X,Y)=H(X)-H(X/Y)={H_X}-{H_X_cond_Y_check}={I_X_Y_check}$
""")

# %%
if (H_X_cond_Y==H_X_cond_Y_check)&(I_X_Y==I_X_Y_check):
    print("Проверка выполнена успешно!")
else:
    print("Ошибка в вычислениях")


