from flask import Flask, render_template, request
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

###VERSION 1.4###


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    number = request.form['number']
    number = int(number) + 1
    number_table = number -1
    return render_template('result.html', number=number, number_table=number_table)

@app.route('/submit_table', methods=['POST'])
def submit_table():
    table_data = {}
        
    # Получаем все значения из формы
    for key in request.form.keys():
        table_data[key] = request.form.getlist(key)

    value_x = float(table_data.get('value_x', [''])[0])
    if 'value_x' in table_data:
        del table_data['value_x']


    # Получаем первые два значения (предполагая, что они всегда есть)
    first_value_table = table_data.get('cell_0_1', [''])[0]
    second_value_table = table_data.get('cell_0_2', [''])[0]

    # Удаляем первые два значения из table_data
    if 'cell_0_1' in table_data:
        del table_data['cell_0_1']
    if 'cell_0_2' in table_data:
        del table_data['cell_0_2']



    # Преобразуем запятые в точки
    table_data = {key: [str(val).replace(',', '.') for val in value] for key, value in table_data.items()}

    # Преобразуем данные в числовой формат
    first_column_x = []
    second_column_y = []
    print("!!!!!!!!!!!!!!!!!!!!!!!", table_data, type(table_data))

    for key, values in table_data.items():
        for value in values:
            if key.endswith('_1'):
                first_column_x.append(value)
            elif key.endswith('_2'):
                second_column_y.append(value)

    # Преобразуем данные в числовой формат
    first_column_x = []
    second_column_y = []
    for key, values in table_data.items():
        for value in values:
            if key.endswith('_1'):
                first_column_x.append(float(value))  # Преобразуем в числовой формат сразу
            elif key.endswith('_2'):
                second_column_y.append(float(value))  # Преобразуем в числовой формат сразу


    print('table_data:', table_data)
    print('first_column_x:', first_column_x)
    print('second_column_y:', second_column_y)
    print('type(first_column_x)',type(first_column_x[0]),first_column_x[0])
    print('type(value_x)',type(value_x),value_x)

    def lineinaya(spisok_x, spisok_y, value_x):
        #Линейная
        print('######################')
        print('Экспонента')
        print()
        try:
            x = np.array(spisok_x)
            y = np.array(spisok_y)
            # Вычисляем коэффициент корреляции
            correlation, p_value = pearsonr(x, y)
            print("Коэффициент корреляции Линейная:", correlation)
            # correlation = round(correlation, 3) 
            

            x = np.array(spisok_x).reshape((-1, 1))
            y = np.array(spisok_y)

            model = LinearRegression()
            model.fit(x, y)
            r_sq = model.score(x, y)
            print('Коэффициент детерминации : Линейная', r_sq)

            # Предсказываем значения для каждого наблюдения
            y_pred = model.predict(x)
            # Рассчитываем среднюю ошибку аппроксимации
            mape = mean_absolute_percentage_error(y, y_pred)
            print("Средняя ошибка аппроксимации: Линейная", mape * 100, "%")

            # Расчет у по x 
            new_x = np.array([[value_x]])
            y_pred = model.predict(new_x)
            print('y предсказанная Линейная при x = :', y_pred)
            fin=[]
            regressiya = 'Линейная'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            
            return  fin
        except:
            print('ошибка расчетов')
            fin = [0, 0, 0]
            return fin

    def eksponenta(spisok_x, spisok_y, value_x):
        #Экспонента
        print('######################')
        print('Экспонента')
        print()
        try:
            x = np.array(spisok_x)
            y = np.array(spisok_y)

            # Преобразование данных для экспоненциальной регрессии
            coefficients = np.polyfit(x, np.log(y), 1)
            y_fit = np.exp(coefficients[1]) * np.exp(coefficients[0] * x)

            # Вычислим коэффициент детерминации
            residuals = y - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            correlation=r_squared**(0.5)

            x_exp = np.exp(x)

            print("Коэффициент детерминации :", r_squared)
            print("индекс корреляции :", correlation)

            # Заданное значение x для предсказания y
            x_pred = value_x

            # Предсказание y для заданного x
            y_pred = np.exp(coefficients[1]) * np.exp(coefficients[0] * x_pred)

            print("Предсказанное значение y для x =", x_pred, ":", y_pred)
            mape = np.mean(np.abs((y - y_fit) / y)) * 100
            print("Средняя ошибка аппроксимации (MAPE):", mape)
            fin = []
            regressiya = 'Экспонента'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            return  fin
        except:
            print('ошибка расчетов')
            fin = [0, 0, 0]
            return fin
    

    def giperbolicheskaya(spisok_x, spisok_y, value_x):
        #### 3 Гиперболическая

        print()
        print('Гиперболическая')
        print()
        try:
            x = np.array(spisok_x)
            y = np.array(spisok_y)

            def hyperbolic_func(x, a, b):
                return a / x + b

            popt, _ = curve_fit(hyperbolic_func, x, y)

            a_opt, b_opt = popt
            

            y_pred = hyperbolic_func(x, a_opt, b_opt)

            r_squared = r2_score(y, y_pred)
            correlation = np.corrcoef(y, y_pred)[0, 1]

            print(f"Коэффициент детерминации: {r_squared}")
            print(f"Индекс корреляции: {correlation}")

            # Вычисление средней ошибки аппроксимации (MAPE)
            mape = mean_absolute_percentage_error(y, y_pred)
            print("Средняя ошибка аппроксимации (MAPE):", mape)
            x = value_x
            y_pred = hyperbolic_func(x, a_opt, b_opt)
            print('y_pred гиперболическая ',y_pred, ' при х = 50')
            fin = []
            regressiya = 'Гиперболическая'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            return  fin
        except:
            print('ошибка расчетов')
            fin = [0, 0, 0]
            return fin



    def stepennaya(spisok_x, spisok_y, value_x):
        #### 4 Степенная
        print()
        print('Степенная')
 
        try:
            # Преобразование данных для степенной регрессии
            # Преобразование в numpy массивы
            x = np.array(spisok_x)
            y = np.array(spisok_y)

            # Преобразование данных для степенной регрессии
            log_x = np.log(x)
            log_y = np.log(y)

            # Выполнение линейной регрессии в преобразованном пространстве
            coefficients = np.polyfit(log_x, log_y, 1)
            a = np.exp(coefficients[1])
            b = coefficients[0]

            # Вычисление y_pred
            y_pred = a * x**b

            # 1. Коэффициент детерминации (R-squared)
            r_squared = r2_score(y, y_pred)
            print("1. Коэффициент детерминации (R-squared):", r_squared)

            # 2. Индекс корреляции
            correlation = np.corrcoef(y, y_pred)[0, 1]
            print("2. Индекс корреляции:", correlation)

            # 3. Средняя ошибка аппроксимации (MAPE)
            def mean_absolute_percentage_error(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = mean_absolute_percentage_error(y, y_pred)
            print("3. Средняя ошибка аппроксимации (MAPE):", mape)

            # 4. Предсказание y для заданного x
            x_pred = 50
            y_pred = a * x_pred**b
            print(f"4. Предсказанное значение y для x = {x_pred}: {y_pred}")
            fin = []
            regressiya = 'Степенная'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            
            
            return  fin
        except:
                print('ошибка расчетов')
                fin = [0, 0, 0]
                return fin



    def parabolicheskaya(spisok_x, spisok_y, value_x):
        #### 5 Параболическая
        print()
        print('Параболическая')
        print()
        try:
            x = np.array(spisok_x)
            y = np.array(spisok_y)
            p = np.polyfit(x, y, 2)  # Подгоняем квадратичную функцию (степень 2) к данным
            y_pred = np.polyval(p, x)  # Предсказанные значения
            
            # Вычисление коэффициента детерминации
            correlation_matrix = np.corrcoef(y, y_pred)
            correlation = correlation_matrix[0, 1]
            r_squared = correlation**2
            print('Коэффициент детерминации (R^2):', r_squared)

            # Вычисление индекса корреляции
            print('Индекс корреляции:', correlation)

            # Вычисление средней ошибки аппроксимации (MAPE)
            mape = mean_absolute_percentage_error(y, y_pred)
            print("Средняя ошибка аппроксимации (MAPE):", mape)

            # Предсказание y для заданного x
            x_pred = value_x
            y_pred = np.polyval(p, x_pred)  # Предсказанное значение
            print(f'y_pred гиперболическая {y_pred} при x = 50')
            print()

            fin = []
            regressiya = 'Параболическая'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            
            return  fin
        except:
            print('ошибка расчетов')
            fin = [0, 0, 0]
            return fin


    
    def logarifmicheskaya(spisok_x, spisok_y, value_x):
        #### Логарифмическая регрессия
        print()
        print('Логарифмическая регрессия')
        print()
        try:
            x = np.array(spisok_x)
            y = np.array(spisok_y)

            # Преобразование данных для логарифмической регрессии
            log_x = np.log(x)
            p = np.polyfit(log_x, y, 1)  # Подгоняем линейную функцию к данным
            a = p[0]
            b = p[1]

            # Вычисление y_pred
            y_pred = a * np.log(x) + b

            # 1. Коэффициент детерминации (R-squared)
            r_squared = r2_score(y, y_pred)
            print("1. Коэффициент детерминации (R-squared):", r_squared)

            # 2. Индекс корреляции
            correlation = np.corrcoef(y, y_pred)[0, 1]
            print("2. Индекс корреляции:", correlation)

            # 3. Средняя ошибка аппроксимации (MAPE)
            def mean_absolute_percentage_error(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = mean_absolute_percentage_error(y, y_pred)
            print("3. Средняя ошибка аппроксимации (MAPE):", mape)

            # 4. Предсказание y для заданного x
            x_pred = value_x
            y_pred = a * np.log(x_pred) + b
            print(f"4. Предсказанное значение y для x = {x_pred}: {y_pred}")

            fin = []
            regressiya = 'Логарифмическая'
            fin.append(correlation)
            fin.append(mape)
            fin.append(y_pred)
            fin.append(regressiya)
            
            return  fin
        except:
            print('ошибка расчетов')
            fin = [0, 0, 0]
            return fin


    fin_lin = lineinaya(first_column_x, second_column_y, value_x)
    fin_eksp = eksponenta(first_column_x, second_column_y, value_x)
    fin_gip = giperbolicheskaya(first_column_x, second_column_y, value_x)
    fin_step = stepennaya(first_column_x, second_column_y, value_x)
    fin_parab = parabolicheskaya(first_column_x, second_column_y, value_x)
    fin_loga = logarifmicheskaya(first_column_x, second_column_y, value_x)


    # Все списки в один
    all_lists = [fin_lin, fin_eksp, fin_gip, fin_step, fin_parab, fin_loga]

    # Интервалы
    intervals = {
        "0.99 до 1": [],
        "0.9 до 0.99": [],
        "0.7 до 0.9": [],
        "0.5 до 0.7": [],
        "0.3 до 0.5": [],
        "0.1 до 0.3": [],
        "0 до 0.1": []
    }

    # Разбиение по интервалам
    for lst in all_lists:
        first_value = lst[0]
        if 0.99 <= first_value <= 1:
            intervals["0.99 до 1"].append(lst)
        elif 0.9 <= first_value < 0.99:
            intervals["0.9 до 0.99"].append(lst)
        elif 0.7 <= first_value < 0.9:
            intervals["0.7 до 0.9"].append(lst)
        elif 0.5 <= first_value < 0.7:
            intervals["0.5 до 0.7"].append(lst)
        elif 0.3 <= first_value < 0.5:
            intervals["0.3 до 0.5"].append(lst)
        elif 0.1 <= first_value < 0.3:
            intervals["0.1 до 0.3"].append(lst)
        elif 0 <= first_value < 0.1:
            intervals["0 до 0.1"].append(lst)

    reg = ''
    #проверяем интервалы 
    def check_inter(interval):
        if "0.99 до 1" in intervals:
            values123 = intervals["0.99 до 1"]
            min_value_list = min(values123, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0.9 до 0.99' in intervals  and intervals['c']:
            values = intervals['0.9 до 0.99']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0.7 до 0.9' in intervals  and intervals['0.7 до 0.9']:
            values = intervals['0.7 до 0.9']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0.5 до 0.7' in intervals  and intervals['0.5 до 0.7']:
            values = intervals['0.5 до 0.7']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0.3 до 0.5' in intervals  and intervals['0.3 до 0.5']:
            values = intervals['0.3 до 0.5']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0.1 до 0.3' in intervals  and intervals['0.1 до 0.3']:
            values = intervals['0.1 до 0.3']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        elif '0 до 0.1' in intervals  and intervals['0 до 0.1']:
            values = intervals['0 до 0.1']
            min_value_list = min(values, key=lambda x: x[1])
            reg = min_value_list[3]
            return min_value_list[2], reg 
        else:
            return print('ошибка')
    


    result, reg = check_inter(intervals)
    # result = round(result[0], 4)
    
    result = float(round(result,4))
    print('check_inter(intervalsresult)',result, type(result))

    
    print("Результат:48" , )
    print('Линейная')
    print(fin_lin,)
    print('эспк')
    print(fin_eksp,)
    print('гип')
    print(fin_gip,)
    print('степ')
    print(fin_step,)
    print('параб')
    print(fin_parab,)
    print('лога')
    print(fin_loga,)
    print('reg', reg)


    return render_template('table_result.html', table_data=table_data, first_value=first_value_table, second_value=second_value_table, result=result, value_x=value_x,reg=reg )



if __name__ == '__main__':
    app.run(debug=True)
