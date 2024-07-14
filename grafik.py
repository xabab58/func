from io import BytesIO
import base64
import matplotlib.pyplot as plt
import os
from threading import Thread
import threading
from io import BytesIO
import base64

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np

matplotlib.use("agg")


def plot_and_encode(first_column_x,second_column_y,funciya, a , b , x,c=0):
    c=c
    e=np.e
    first_column_x_x=[]
    
    
    for i in first_column_x:
        first_column_x_x.append(i)
    first_column_x_x.append(x)    
    y_pred=[]
    if funciya == 'Линейная':
        for i in first_column_x_x:
            y=a*i+b
            y_pred.append(y)
    elif funciya == 'Экспонента':
        for i in first_column_x_x:
            y= b * (e ** (a * i))
            y_pred.append(y)
    elif funciya == 'Гиперболическая':
        for i in first_column_x_x:
            y= (a / i) + b
            y_pred.append(y)
    elif funciya == 'Степенная':
        for i in first_column_x_x:
            y= (a * i) ** b
            y_pred.append(y)
    elif funciya == 'Параболическая':
        for i in first_column_x_x:
            y = a * (float(i) ** 2) + b * i + c
            y_pred.append(y)
    elif funciya == 'Логарифмическая':
        for i in first_column_x_x:
            y= a * np.log(i) + b
            y_pred.append(y)

    print("first_column_x=",first_column_x)
    print("second_column_y=",second_column_y)
    print("first_column_x_x=",first_column_x_x)
    print("y_pred=",y_pred)
            

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    

    print('first_column_x= ',first_column_x, 'second_column_y= ',second_column_y, 'a=', a, 'b=',b, 'funciya=',funciya,'y_pred=',y_pred,'first_column_x=',first_column_x,)
    # Создание фигуры
    plt.figure(figsize=(10, 6))
    
    # Построение второго графика
    plt.plot(first_column_x_x, y_pred, 's-', label='Предсказанный ')
    # Построение первого графика
    plt.plot(first_column_x, second_column_y, 'o-', label='Исходный график')
    
    # Добавление легенды
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return plot_data



