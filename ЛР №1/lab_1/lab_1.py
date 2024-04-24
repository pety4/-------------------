import pandas as pd
import numpy as np
import fileinput
import matplotlib.pyplot as plt
from math import ceil
from tkinter import *

#Значения t-распределения Стьюдента для доверительной вероятности Р=0,95
#и данном числе степеней свободы f
t={1:12.71,
   2:4.30,
   3:3.18,
   4:2.78,
   5:2.57,
   6:2.45,
   7:2.36,
   8:2.31,
   9:2.26,
   10:2.23,
   11:2.20,
   12:2.18,
   13:2.16,
   14:2.14,
   15:2.13,
   16:2.12,
   17:2.11,
   18:2.10,
   19:2.09,
   20:2.09,
   21:2.08,
   22:2.07,
   23:2.07,
   24:2.06,
   25:2.06,
   26:2.06,
   27:2.05,
   28:2.05}

z=1.96
#желаемый показатель точности
E_ideal=4.8

#Фамилия экспериментатора
experimenter_last_name='Orlov'
#experimenter_last_name='Dementeva'
#experimenter_last_name='Mylnikov'

#данные об экспериментах
exp_1={'13':['1','3','6','9','13'],
    '17':['1','3','6','9','13','17']}

#Орлов
exp_2={'13':['500','450','400','350'],
           '17':['550','500','450','400']}


#Мыльников
#exp_2={'13':['500','450','440','390','340','290'],
#           '17':['500','480','430','380','330']}


#функция очистки данных от выбросов и ложных нажатий
def clear_data(matrix,log):
    wrong_measurement=[]
    for measurement in log:
        if (measurement[4]==1) or (measurement[4]==3):
            matrix[int(measurement[1])-1,int(measurement[2])-1]-=1
            wrong_measurement.append(int(measurement[0])-1)
    log=np.delete(log,wrong_measurement,axis=0)
    return matrix, log

#функция оформления графика
def plot_design():
    plt.xlabel("I, бит",
                fontsize=14,
                fontfamily='Times New Roman')
    plt.xticks(fontsize=14,
               fontfamily='Times New Roman')
    plt.ylabel("ВР, с",
               rotation=0,
               loc='top',
               fontsize=14,
               fontfamily='Times New Roman')
    plt.yticks(fontsize=14,
               fontfamily='Times New Roman')
    plt.legend()
    plt.grid()
    return

#функция для вычисления логарифма
def log_2(x):
    if x==0:
        return 0
    else: 
        return np.log2(x)
    
#функция расчёта H(X)
def H_X_count(matrix):
    N=matrix.sum()
    joint_probabilities_matrix=matrix/N
    if len(matrix.shape)==1:
        P_x=joint_probabilities_matrix.sum()
        H_X=P_x*log_2(P_x)
        H_X=np.round(H_X,6)
        return H_X
    n,m=matrix.shape
    P_x=np.zeros(n)
    for i in range(n):
        for j in range(m):
            P_x[i]+=joint_probabilities_matrix[i][j]
    H_X=0.0
    for i in range(n):
        H_X-=P_x[i]*log_2(P_x[i])
    H_X=np.round(H_X,6)
    return H_X

#функция расчёта H(Y)
def H_Y_count(matrix):
    N=matrix.sum()
    joint_probabilities_matrix=matrix/N
    if len(matrix.shape)==1:
        P_x=joint_probabilities_matrix.sum()
        H_X=P_x*log_2(P_x)
        H_X=np.round(H_X,6)
        return H_X
    n,m=matrix.shape
    p_y=np.zeros(m)
    for j in range(m):
        for i in range(n):
            p_y[j]+=joint_probabilities_matrix[i][j]
    H_Y=0.0
    for j in range(m):
        H_Y-=p_y[j]*log_2(p_y[j])
    H_Y=np.round(H_Y,6)
    return H_Y

#функция расчёта H(X,Y)
def H_X_Y_count(matrix):
    N=matrix.sum()
    joint_probabilities_matrix=matrix/N
    H_X_Y=0.0
    n,m=matrix.shape
    for i in range(n):
        for j in range(m):
            H_X_Y-=joint_probabilities_matrix[i][j]*log_2(joint_probabilities_matrix[i][j])
    H_X_Y=np.round(H_X_Y,6)
    return H_X_Y

#функция расчёта I(X,Y)
def I_count(matrix):
    H_X=H_X_count(matrix)
    H_Y=H_Y_count(matrix)
    H_X_Y=H_X_Y_count(matrix)
    I_X_Y=np.round(H_X+H_Y-H_X_Y,6)
    return I_X_Y

#функция построения диаграммы информационного канала
def info_channel_diagram_count(matrix):
    H_X=H_X_count(matrix)
    H_Y=H_Y_count(matrix)
    I_X_Y=I_count(matrix)
    H_X_cond_Y=np.round(H_X-I_X_Y,6)
    H_Y_cond_X=np.round(H_Y-I_X_Y,6)
    return H_X, H_X_cond_Y, I_X_Y, H_Y_cond_X, H_Y

#функция форматирования файла
def file_formatting(filename):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('	', ','), end='')
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('#', ''), end='')

#функция расчёта мат. ожидания
def M_x_count(log):
    n=len(log)
    M_x=0
    for i in range(n):
        M_x+=log[i]
    M_x/=n
    return M_x

#функция расчёта СКО результатов наблюдения
def SKO_count(log):
    M_x=M_x_count(log)
    n=len(log)
    SKO=0
    for i in range(n):
        SKO+=np.power(log[i]-M_x,2)
    if SKO==0:
        return SKO
    SKO=np.sqrt(SKO/(n-1))
    return SKO

#функция расчёта СКО результатов измерений
def SKO_x_count(log):
    n=len(log)
    SKO=SKO_count(log)
    SKO_x=SKO/np.sqrt(n)
    return SKO_x

#функция расчёта доверительного интервала
def confidence_interval_count(log):
    n=len(log)
    SKO_x=SKO_x_count(log)
    f=n-1
    epsilon=t[f]*SKO_x
    return epsilon

#функция расчёта меры изменчивости
def measure_of_variability_count(log):
    SKO=SKO_count(log)
    M_x=M_x_count(log)
    v=SKO/M_x*100
    return v

#функция расчёта достаточного количества опытов
def sufficient_experiments_count(z,E,log):
    v=measure_of_variability_count(log)
    n=np.power(z*v/E,2)
    return n

#функция расчёта показателя точности
def accuracy_rate_count(z,log):
    v=measure_of_variability_count(log)
    n=len(log)
    E=v*np.sqrt(np.power(z,2)/n)
    return E

#функция построения доверительного интервала
def confidence_interval_plot(I,t,confidence_interval, color='#2187bb'):
    horizontal_line_width=0.22
    left = I - horizontal_line_width / 2
    top = t - confidence_interval
    right = I + horizontal_line_width / 2
    bottom = t + confidence_interval
    plt.plot([I, I], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(I, t, 'o', color='#f44336')
    return

#функция расчёта выборочного стандартного отклонения
def selective_standard_deviation_count(I,reaction_time,a,b):
    n=len(reaction_time)
    sigma_T=0
    for i in range(n):
        T=a+b*I[i]
        sigma_T+=np.power(reaction_time[i]-T,2)
    sigma_T=np.sqrt(sigma_T/(n-2))
    return sigma_T

#функция построения линейной регрессии МНК с трубкой точности
def linear_regression_LSM_plot(I,reaction_time, precision_tube=True):
    M_I=M_x_count(I)
    M_t=M_x_count(reaction_time)
    a=0
    b=0
    delta=0
    n=len(reaction_time)
    for i in range(n):
        delta+=np.power(I[i]-M_I,2)
        b+=(I[i]-M_I)*reaction_time[i]
    b/=delta
    a=M_t-b*M_I
    plt.plot([I[0],I[-1]],
             [a+b*I[0],a+b*I[-1]],
             color='g',
             label='МНК')
    
    #построение трубки точности
    if precision_tube==True:
        sigma_T=selective_standard_deviation_count(I,reaction_time,a,b)
        eps_a=sigma_T*t[n-2]*np.sqrt(1/n+np.power(M_I,2)/delta)
        eps_b=sigma_T*t[n-2]*np.sqrt(1/delta)
        eps_T=[]
        for I_0 in I:
            eps_T.append(sigma_T*t[n-2]*np.sqrt(1/n+np.power(I_0-M_I,2)/delta))

        plt.plot([I[0],I[-1]],
                [(a+eps_a)+(b+eps_b)*I[0],(a+eps_a)+(b+eps_b)*I[-1]],
                color='c',
                label='Трубка точности')
        plt.plot([I[0],I[-1]],
                [(a-eps_a)+(b-eps_b)*I[0],(a-eps_a)+(b-eps_b)*I[-1]],
                color='c')
        return a, b, eps_a, eps_b, eps_T
    return a,b

#функция расчёта взвешенного среднего
def weighted_average_count(arr,w):
    M=0
    w=np.array(w)
    for i in range(len(arr)):
       M+=w[i]*arr[i]
    M/=w.sum()
    return M

#функция построения линейной регрессии взвешенным МНК
def linear_regression_weighted_LSM_plot(I,reaction_time,log):
    w=[]
    for i in range(len(I)):  
        SKO_x=SKO_x_count(log[i][:, 3])
        w.append(1/np.power(SKO_x,2))
    b=0
    delta=0
    M_I=weighted_average_count(I,w)
    M_t=weighted_average_count(reaction_time,w)
    for i in range(len(I)):
        delta+=w[i]*np.power(I[i]-M_I,2)
        b+=w[i]*(I[i]-M_I)*reaction_time[i]
    b/=delta
    a=M_t-b*M_I
    plt.plot([I[0],I[-1]],
             [a+b*I[0],a+b*I[-1]],
             color='m',
             label='Взвешенный МНК')
    return a, b

#функция обработки результатов эксперимента №1
def processing_experiment_1_results(key_num,keyboard_num,log_file_name, matrix_file_name):
    #форматируем файлы
    for log in log_file_name:
        file_formatting(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_1/{key_num}_keys/{log}')
    for matrix in matrix_file_name:
        file_formatting(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_1/{key_num}_keys/{matrix}')

    #считываем данные с файлов и убираем выбросы и неверные нажатия
    log=[]
    for file_name in log_file_name:
        log.append(np.loadtxt(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_1/{key_num}_keys/{file_name}', delimiter=","))
    matrix=[]
    for file_name in matrix_file_name:
        matrix.append(np.loadtxt(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_1/{key_num}_keys/{file_name}', delimiter=","))
    for i in range(len(matrix)):
        matrix[i],log[i]=clear_data(matrix[i],log[i])
    
    plt.figure(f'Эксперимент №1 ({key_num} клавиш) ({experimenter_last_name})')    
    
    #рассчитываем:
    n=[]
    H_X=[]    
    N=[]
    E_real=[]
    reaction_time_SKO=[]
    v=[]
    for l in log:
        n.append(len(l))#объём выборки
        N.append(ceil(sufficient_experiments_count(z,E_ideal,l[:, 3])))#достаточное количество опытов
        E_real.append(accuracy_rate_count(z,l[:, 3]))#показатель точности
        reaction_time_SKO.append(SKO_count(l[:, 3]))#СКО времени реакции
        v.append(measure_of_variability_count(l[:, 3]))#меры изменчивости


    #рассчитываем количество информации
    for m in matrix:
        H_X.append(H_X_count(m))

    #строим доверительный интервал
    eps=[]
    reaction_time=[]
    for l in log:
        eps.append(confidence_interval_count(l[:, 3]))
        reaction_time.append(M_x_count(l[:, 3]))
    for i in range(len(H_X)):
        confidence_interval_plot(H_X[i],reaction_time[i],eps[i])
     
    #строим прямые линейных регрессий
    a,b,eps_a,eps_b,eps_T=linear_regression_LSM_plot(H_X,reaction_time)
    a_w, b_w=linear_regression_weighted_LSM_plot(H_X,reaction_time,log)
    plot_design()

    #выводим результаты
    print(f'\nРезультаты обработки эксперимента №1 ({key_num} клавиш)\n'
          f'Объём выборки для анализа:\n'
          f'{n}\n'
          f'Количество информации I, бит:\n'
          f'{np.round(H_X,3)}\n'
          f'Среднее время реакции, мс:\n'
          f'{np.round(reaction_time,2)}\n'
          f'СКО времени реакции, мс:\n'
          f'{np.round(reaction_time_SKO,2)}\n'
          f'Доверительный интервал, мс:\n'
          f'{np.round(eps,2)}\n'
          f'Мера изменчивости, %:\n'
          f'{np.round(v,2)}\n'
          f'Показатель точности E, %:\n'
          f'{np.round(E_real,2)}\n'
          f'Достаточное количества опытов (E={E_ideal}%):\n'
          f'{N}\n'
          f'Параметры закона Хика: T=a+bI\n'
          f'\nНевзвешенный метод:\n'
          f'a={np.round(a,2)}+-{np.round(eps_a,2)} мс, b={np.round(b,2)}+-{np.round(eps_b,2)} мс/бит\n'
          f'Скорость передачи информации: {np.round(1000/b,2)} бит/с\n'
          f'Латентный период: {np.round(a,2)} мс\n'
          f'Доверительные интервалы для ВР, мс:\n'
          f'{np.round(eps_T,2)}\n'
          f'\nВзвешенный метод:\n'
          f'a={np.round(a_w,2)} мс, b={np.round(b_w,2)} мс/бит\n'
          f'Скорость передачи информации: {np.round(1000/b_w,2)} бит/с\n'
          f'Латентный период: {np.round(a_w,2)} мс\n'
          )
    return

#функция построения точки на графике
def dot_plot(I,t,color='#f44336'):
    plt.plot(I, t, 'o', color=color)
    return

#функция построения диаграммы информационного канала
def info_channel_diagram_window_create(info_channel_diagram, exposure_time, keyboard_num):
    font_style=("Times New Roman",12)
    H_X,H_X_cond_Y,I_X_Y,H_Y_cond_X,H_Y=info_channel_diagram
    info_channel_diagram_window=Tk()
    info_channel_diagram_image = PhotoImage(file="info_channel_diagram.png")
    m=info_channel_diagram_image.width()
    n=info_channel_diagram_image.height()
    info_channel_diagram_window.title(f'Диаграмма информационного канала')
    info_channel_diagram_window.geometry(f'{m}x{n}')

    canvas = Canvas(info_channel_diagram_window,bg="white", width=m, height=n)
    canvas.pack(anchor=CENTER, expand=1)
 
    info_channel_diagram_image = PhotoImage(file="info_channel_diagram.png")
 
    canvas.create_image(0,0,anchor=NW, image=info_channel_diagram_image)
    
    canvas.create_text(2, 2,
                       anchor=NW,
                       text=f'Клавиатура №{keyboard_num}\n'
                       f'Экспозиция: {exposure_time} мс',
                       font=font_style)
    canvas.create_text(20, (n-65)/2,
                       anchor=NW,
                       text=f'H(X)={np.round(H_X,4)} бит',
                       font=font_style)
    canvas.create_text((m-150)/2, (n-65)/2,
                       anchor=NW,
                       text=f'I(X,Y)={np.round(I_X_Y,4)} бит',
                       font=font_style)
    canvas.create_text((m-25)/2, (n-250)/2,
                       anchor=NW,
                       text=f'H(X/Y)={np.round(H_X_cond_Y,4)} бит',
                       font=font_style)
    canvas.create_text((m+25)/2, (n+250)/2,
                        anchor=SE,
                        text=f'H(Y/X)={np.round(H_Y_cond_X,4)} бит',
                        font=font_style)
    canvas.create_text(m-20, (n-65)/2,
                       anchor=NE,
                       text=f'H(Y)={np.round(H_Y,4)} бит',
                       font=font_style)
    info_channel_diagram_window.mainloop()



def processing_experiment_2_results(key_num,keyboard_num,log_file_name, matrix_file_name):
    #форматируем файлы
    for log in log_file_name:
        file_formatting(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_2/{key_num}_keys/{log}')
    for matrix in matrix_file_name:
        file_formatting(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_2/{key_num}_keys/{matrix}')

    #считываем данные с файлов
    log=[]
    for file_name in log_file_name:
        log.append(np.loadtxt(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_2/{key_num}_keys/{file_name}', delimiter=","))
    matrix=[]
    for file_name in matrix_file_name:
        matrix.append(np.loadtxt(f'{experimenter_last_name}/keyboard_{keyboard_num}/data_exp_2/{key_num}_keys/{file_name}', delimiter=","))

    plt.figure(f'Эксперимент №2 ({key_num} клавиш) ({experimenter_last_name})')

    #строим диаграмму информационного канала
    info_channel_diagram=[]
    I_X_Y=[]
    for i in range(len(matrix)):
        info_channel_diagram.append(info_channel_diagram_count(matrix[i]))
        I_X_Y.append(info_channel_diagram[-1][2])

    #строим точки на графике
    exposure_time=[]
    for time in exp_2[f'{key_num}']:
        exposure_time.append(int(time))
    
    for i in range(len(I_X_Y)):
        dot_plot(I_X_Y[i], exposure_time[i])

    #строим прямые линейных регрессий
    a,b=linear_regression_LSM_plot(I_X_Y,exposure_time,precision_tube=False)
    a_w, b_w=linear_regression_weighted_LSM_plot(I_X_Y,exposure_time,log)
    plot_design()

    #выводим результаты
    info_channel_diagram_string=''
    for i in range(len(info_channel_diagram)):
        info_channel_diagram_window_create(info_channel_diagram[i],exp_2[f'{key_num}'][i],keyboard_num)
        info_channel_diagram_string+=(f'\nВремя экспозиции: {exp_2[f'{key_num}'][i]} мс\n')
        info_channel_diagram_string+=(f'H(X)={np.round(info_channel_diagram[i][0],4)} бит\n'
                                        f'H(X/Y)={np.round(info_channel_diagram[i][1],4)} бит\n'
                                        f'I(X,Y)={np.round(info_channel_diagram[i][2],4)} бит\n'
                                        f'H(Y/X)={np.round(info_channel_diagram[i][3],4)} бит\n'
                                        f'H(Y)={np.round(info_channel_diagram[i][4],4)} бит\n')
    
    print(f'\nРезультаты обработки эксперимента №2 ({key_num} клавиш)\n'
          f'Диаграммы информационного канала:\n'
          f'{info_channel_diagram_string}\n'
          f'Параметры закона Хика: T=a+bI\n'
          f'\nНевзвешенный метод:\n'
          f'a={np.round(a,2)} мс, b={np.round(b,2)} мс/бит\n'
          f'Скорость передачи информации: {np.round(1000/b,2)} бит/с\n'
          f'Латентный период: {np.round(a,2)} мс\n'
          f'\nВзвешенный метод:\n'
          f'a={np.round(a_w,2)} мс, b={np.round(b_w,2)} мс/бит\n'
          f'Скорость передачи информации: {np.round(1000/b_w,2)} бит/с\n'
          f'Латентный период: {np.round(a_w,2)} мс\n'
          )
    return

#основная программа
if __name__ == "__main__":
    log_file_name=[[],[]]
    matrix_file_name=[[],[]]
    
    for i in exp_1['13']:
        log_file_name[0].append(f'log_{13}_keys_{i}_stim.csv')
        matrix_file_name[0].append(f'matrix_{13}_keys_{i}_stim.csv')
    for i in exp_1['17']:
        log_file_name[1].append(f'log_{17}_keys_{i}_stim.csv')
        matrix_file_name[1].append(f'matrix_{17}_keys_{i}_stim.csv')
    processing_experiment_1_results(13,1,log_file_name[0],matrix_file_name[0])
    processing_experiment_1_results(17,1,log_file_name[1],matrix_file_name[1])

    log_file_name=[[],[]]
    matrix_file_name=[[],[]]
    for i in exp_2['13']:
        log_file_name[0].append(f'log_{13}_keys_{i}_ms.csv')
        matrix_file_name[0].append(f'matrix_{13}_keys_{i}_ms.csv')
    for i in exp_2['17']:
        log_file_name[1].append(f'log_{17}_keys_{i}_ms.csv')
        matrix_file_name[1].append(f'matrix_{17}_keys_{i}_ms.csv')


    processing_experiment_2_results(13,1,log_file_name[0], matrix_file_name[0])
    processing_experiment_2_results(17,1,log_file_name[1], matrix_file_name[1])
    plt.show()
