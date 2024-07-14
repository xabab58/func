import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_percentage_error



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

        # Коэффициенты a и b
        a = float(model.coef_[0])
        b = float(model.intercept_)
        print(f'Коэффициенты Линейная: a = {a}, b = {b}')

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
        # correlation = round(correlation[0], 4)
        
        correlation=float(correlation)
        y_pred=float(y_pred)
        y_pred=round(y_pred,4)
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred)
        fin.append(regressiya)
        fin.append(a)  # Добавляем коэффициент a
        fin.append(b)  # Добавляем коэффициент b
        
        return  fin
    except:
        fin = [0, 0, 0, 'ошибка расчетов']
        return fin

def eksponenta(spisok_x, spisok_y, value_x):
    # Экспонента
    print('######################')
    print('Экспонента')
    print()
    try:
        x = np.array(spisok_x)
        y = np.array(spisok_y)

        # Проверка на положительность y
        if np.any(y <= 0):
            raise ValueError("Все значения y должны быть положительными для логарифмирования.")
        
        # Преобразование данных для экспоненциальной регрессии
        log_y = np.log(y)
        coefficients = np.polyfit(x, log_y, 1)
        a = coefficients[0]
        b = np.exp(coefficients[1])

        # Построение экспоненциальной регрессии
        y_fit = b * np.exp(a * x)

        # Вычислим коэффициент детерминации
        residuals = y - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        correlation = r_squared**0.5

        print("Коэффициент детерминации:", r_squared)
        print("Индекс корреляции:", correlation)
        print("Коэффициент a:", a)
        print("Коэффициент b:", b)

        # Заданное значение x для предсказания y
        x_pred = value_x

        # Предсказание y для заданного x
        y_pred = b * np.exp(a * x_pred)
        print("Предсказанное значение y для x =", x_pred, ":", y_pred)

        mape = np.mean(np.abs((y - y_fit) / y)) * 100
        print("Средняя ошибка аппроксимации (MAPE):", mape)

        y_pred = float(y_pred)
        y_pred = round(y_pred, 4)

        fin = []
        regressiya = 'Экспонента'
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred)
        fin.append(regressiya)
        fin.append(a)
        fin.append(b)
        return fin
    except Exception as e:
        print("Ошибка:", e)
        fin = [0, 0, 0, 'ошибка расчетов', 0, 0]
        return fin


def giperbolicheskaya(spisok_x, spisok_y, value_x):
    print()
    print('Гиперболическая')
    print()
    try:
        x = np.array(spisok_x)
        y = np.array(spisok_y)

        # Определение гиперболической функции
        def hyperbolic_func(x, a, b):
            return a / x + b

        # Найти оптимальные значения коэффициентов a и b
        popt, _ = curve_fit(hyperbolic_func, x, y)

        a_opt, b_opt = popt
        
        # Предсказанные значения y
        y_pred = hyperbolic_func(x, a_opt, b_opt)

        # Вычисление коэффициента детерминации
        r_squared = r2_score(y, y_pred)
        correlation = np.corrcoef(y, y_pred)[0, 1]

        print(f"Коэффициент детерминации: {r_squared}")
        print(f"Индекс корреляции: {correlation}")

        # Вычисление средней ошибки аппроксимации (MAPE)
        mape = mean_absolute_percentage_error(y, y_pred)
        print("Средняя ошибка аппроксимации (MAPE):", mape)

        # Предсказание y для заданного значения x
        y_pred = hyperbolic_func(value_x, a_opt, b_opt)
        print('y_pred гиперболическая ', y_pred, ' при x =', value_x)
        
        y_pred = float(y_pred)
        y_pred = round(y_pred, 4)
        
        fin = []
        regressiya = 'Гиперболическая'
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred)
        fin.append(regressiya)
        fin.append(a_opt)
        fin.append(b_opt)
        
        return fin
    except Exception as e:
        print("Ошибка:", e)
        fin = [0, 0, 0, 'ошибка расчетов']
        return fin




def stepennaya(spisok_x, spisok_y, value_x):
    #### 4 Степенная
    print()
    print('Степенная')

    try:
        # Преобразование данных для степенной регрессии
        x = np.array(spisok_x)
        y = np.array(spisok_y)

        # Преобразование данных для степенной регрессии
        log_x = np.log(x)
        log_y = np.log(y)

        # Выполнение линейной регрессии в преобразованном пространстве
        coefficients = np.polyfit(log_x, log_y, 1)
        b = coefficients[0]
        log_a = coefficients[1]
        a = np.exp(log_a)

        # Вычисление y_pred на основе исходных данных x
        y_pred = a * x**b

        # Вычисление коэффициента детерминации
        r_squared = r2_score(y, y_pred)
        correlation = np.corrcoef(y, y_pred)[0, 1]

        print(f"Коэффициент детерминации: {r_squared}")
        print(f"Индекс корреляции: {correlation}")

        # Вычисление средней ошибки аппроксимации (MAPE)
        mape = mean_absolute_percentage_error(y, y_pred)
        print("Средняя ошибка аппроксимации (MAPE):", mape)

        # Предсказание y для заданного значения x
        y_pred_value_x = a * value_x**b
        print(f"Предсказанное значение y для x = {value_x}: {y_pred_value_x}")
        
        y_pred_value_x = float(y_pred_value_x)
        y_pred_value_x = round(y_pred_value_x, 4)
        
        fin = []
        regressiya = 'Степенная'
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred_value_x)
        fin.append(regressiya)
        fin.append(a)
        fin.append(b)
        
        return fin
    except Exception as e:
        print("Ошибка:", e)
        fin = [0, 0, 0, 'ошибка расчетов']
        return fin



def parabolicheskaya(spisok_x, spisok_y, value_x):
    #### 5 Параболическая
    print()
    print('Параболическая')
    print()
    try:
        x = np.array(spisok_x)
        y = np.array(spisok_y)
        
        # Подгоняем квадратичную функцию (степень 2) к данным
        p = np.polyfit(x, y, 2)  # p содержит коэффициенты [a, b, c] для функции ax^2 + bx + c
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
        y_pred_value_x = np.polyval(p, value_x)  # Предсказанное значение
        print(f'y_pred параболическая {y_pred_value_x} при x = {value_x}')
        print()
        y_pred_value_x = float(y_pred_value_x)
        y_pred_value_x = round(y_pred_value_x, 4)

        # Вывод коэффициентов a, b и c
        a, b, c = p
        print(f'Коэффициенты: a = {a}, b = {b}, c = {c}')

        fin = []
        regressiya = 'Параболическая'
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred_value_x)
        fin.append(regressiya)
        fin.append(a)
        fin.append(b)
        fin.append(c)
        
        return fin
    except Exception as e:
        print("Ошибка:", e)
        fin = [0, 0, 0, 'ошибка расчетов']
        return fin




def mean_absolute_percentage_error(y_true, y_pred):
    """Вычисление средней ошибки аппроксимации (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
        mape = mean_absolute_percentage_error(y, y_pred)
        print("3. Средняя ошибка аппроксимации (MAPE):", mape)

        # 4. Предсказание y для заданного x
        y_pred_value_x = a * np.log(value_x) + b
        print(f"4. Предсказанное значение y для x = {value_x}: {y_pred_value_x}")
        y_pred_value_x = float(y_pred_value_x)
        y_pred_value_x = round(y_pred_value_x, 4)

        fin = []
        regressiya = 'Логарифмическая'
        fin.append(correlation)
        fin.append(mape)
        fin.append(y_pred_value_x)
        fin.append(regressiya)
        fin.append(a)  # Коэффициент a
        fin.append(b)  # Коэффициент b
        
        return fin
    except Exception as e:
        print("Ошибка:", e)
        fin = [0, 0, 0, 'ошибка расчетов']
        return fin
