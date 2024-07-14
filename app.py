from flask import Flask, render_template, request
from functions import *
from grafik import *



###VERSION 1.5###
PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

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



    print("intervalintervalinterval",intervals)
    reg = ''
    #проверяем интервалы 
    def check_inter(interval):
        try:
            # Проверяем, если есть хотя бы один элемент в значении для каждого интервала
            if "0.99 до 1" in interval and interval["0.99 до 1"]:
                values = interval["0.99 до 1"]
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg 

            if '0.9 до 0.99' in interval and interval['0.9 до 0.99']:
                values = interval['0.9 до 0.99']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg

            if '0.7 до 0.9' in interval and interval['0.7 до 0.9']:
                values = interval['0.7 до 0.9']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg

            if '0.5 до 0.7' in interval and interval['0.5 до 0.7']:
                values = interval['0.5 до 0.7']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg 

            if '0.3 до 0.5' in interval and interval['0.3 до 0.5']:
                values = interval['0.3 до 0.5']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg 

            if '0.1 до 0.3' in interval and interval['0.1 до 0.3']:
                values = interval['0.1 до 0.3']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg 

            if '0 до 0.1' in interval and interval['0 до 0.1']:
                values = interval['0 до 0.1']
                min_value_list = min(values, key=lambda x: x[1])
                reg = min_value_list[3]
                return min_value_list, reg 

            # Если ни один из интервалов не содержит данных
            return 0, 'ошибка'

        except Exception as e:
            print("Ошибка:", e)
            return 0, 'ошибка'
    

    
    result, reg = check_inter(intervals)
    print("result=",result,"reg=",reg)
    

    
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
   

    print('!!!!!!!!!!!!!!!!!234234!!!!!!!!!!!!!!',result)       
    # plot_data = plot_and_encode(first_column_x,second_column_y,result[3],result[4],result[5],value_x)
    if reg =="Параболическая":
        plot_data = plot_and_encode(first_column_x,second_column_y,result[3],result[4],result[5],value_x,result[6],)
    else:
        plot_data = plot_and_encode(first_column_x,second_column_y,result[3],result[4],result[5],value_x)


    return render_template('table_result.html', table_data=table_data, first_value=first_value_table, second_value=second_value_table, result=result[2], value_x=value_x,reg=reg,user_image=plot_data )



if __name__ == '__main__':
    app.run(debug=True)
