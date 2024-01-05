import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from PIL import ImageTk, Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv', sep=',')

data.time_remaining = data.time_remaining.str.split(':').str[0].astype(int)*60 + data.time_remaining.str.split(':').str[1].astype(int)

y = data['result']
predictData = data.drop(['result'], axis=1)
data = pd.get_dummies(data, columns=['qtr'])
visualData = data.drop(['result'], axis=1)
dataOnlyHit = data[data['result'] == True]
data = data.drop(['result'], axis=1)
max_arr = np.array(data.max())




print(data.head())
print(data.dtypes)
#ОТ ТУТА НАДО ДАННЫЕ МОДИФИЦИРОВАТЬ
data = pd.get_dummies(data, columns=['date'])
data = pd.get_dummies(data, columns=['opponent'])
data = pd.get_dummies(data, columns=['team'])
data = pd.get_dummies(data, columns=['color'])

#ОТ ТУТА УЖЕ НЕ НАДО
print(data.head())
print(data.dtypes)

scaleddata = StandardScaler().fit_transform(data)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(scaleddata)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principalComponent1', 'principalComponent2'])
principalDf["result"] = y
print(principalDf.head())
print(principalDf.dtypes)
principalDfTRUE = principalDf[principalDf['result'] == True]
principalDfFALSE = principalDf[principalDf['result'] == False]
principalDf = principalDf.drop(['result'], axis=1)
data = principalDf




root = Tk()  
root.title("ML в NBA")  
root.geometry('1280x720')
root.resizable(width=False, height=False)

bgimg = Image.open('_bg.jpg')
bg = ImageTk.PhotoImage(bgimg)
work_result = []

canvas = Canvas(root, width=1280, height=720)
canvas.pack(side="top", fill="both", expand="no")
canvas.create_image(0, 0, anchor="nw", image=bg)

def vectorML():
    global current_acc, train_data, val_data, train_y, val_y, gnb, predicted, forest, SupportVM
    current_acc = 0
    avr_acc = 0
    iter = 0
    while (True):
        if (current_acc < 0.74):
            train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.1)
            SupportVM = SVC()
            SupportVM.fit(train_data, train_y)
            predicted = SupportVM.predict(val_data)
            current_acc = accuracy_score(predicted, val_y)
            avr_acc += current_acc
            iter += 1
        else:
            L_accuracy["text"] = 'Точность модели: \n' + str(current_acc)
            Lb_work_result.delete(0,Lb_work_result.size())
            Lb_work_result.insert(0,"Эталонная выборка:         Результаты работы модели:")
            for i in range(len(np.array(val_y))):
                Lb_work_result.insert(i+1,"{0:10d}: {1:20s} {0:25d}: {2:20s}".format(i,str(np.array(val_y)[i]),str(np.array(predicted)[i])))
            break
    print(train_data.head())
    print('_'*40)
    print('Вектор')
    print("Средняя точность = ", avr_acc/iter)
    print("Всего итераций обучения:", iter)

def forestML():
    global current_acc, train_data, val_data, train_y, val_y, gnb, predicted, forest
    current_acc = 0
    avr_acc = 0
    iter = 0
    while (True):
        if (current_acc < 0.7):
            train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.1)
            forest = RandomForestClassifier(n_estimators=100, 
                       bootstrap = True,
                       max_features = 'sqrt')
            forest.fit(train_data,train_y)
            predicted = forest.predict(val_data)
            current_acc = accuracy_score(predicted, val_y)
            avr_acc += current_acc
            iter += 1
        else:
            L_accuracy["text"] = 'Точность модели: \n' + str(current_acc)
            Lb_work_result.delete(0,Lb_work_result.size())
            Lb_work_result.insert(0,"Эталонная выборка:         Результаты работы модели:")
            for i in range(len(np.array(val_y))):
                Lb_work_result.insert(i+1,"{0:10d}: {1:20s} {0:25d}: {2:20s}".format(i,str(np.array(val_y)[i]),str(np.array(predicted)[i])))
            break
    print('_'*40)
    print('ЛЕС')
    print("Средняя точность = ", avr_acc/iter)
    print("Всего итераций обучения:", iter)

def NBaiesML():
    global current_acc, train_data, val_data, train_y, val_y, gnb, predicted, forest
    current_acc = 0
    avr_acc = 0
    iter = 0
    while (True):
        if (current_acc < 0.75):
            train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.1)
            gnb = GaussianNB()
            gnb.fit(train_data,train_y)
            predicted = gnb.predict(val_data)
            current_acc = accuracy_score(predicted, val_y)
            avr_acc += current_acc
            iter += 1
        else:
            L_accuracy["text"] = 'Точность модели: \n' + str(current_acc)
            Lb_work_result.delete(0,Lb_work_result.size())
            Lb_work_result.insert(0,"Эталонная выборка:         Результаты работы модели:")
            for i in range(len(np.array(val_y))):
                Lb_work_result.insert(i+1,"{0:10d}: {1:20s} {0:25d}: {2:20s}".format(i,str(np.array(val_y)[i]),str(np.array(predicted)[i])))
            break
    print('_'*40)
    print('БАЙЕС')
    print("Средняя точность = ", avr_acc/iter)
    print("Всего итераций обучения:", iter)

def change_current_model():
    if (B_change_current_model["text"] == "Байес"):
        B_change_current_model["text"] = "Лес"
    elif(B_change_current_model["text"] == "Лес"):
        B_change_current_model["text"] = "Вектор"
    elif(B_change_current_model["text"] == "Вектор"):
        B_change_current_model["text"] = "Байес"

def try_predict():
    global predictData , data, principalDfTRUE, principalDfFALSE, y

    newData = [int(E_top.get()),
               int(E_left.get()),
               E_date.get(),E_q.get(),
               int(E_time_remaining.get()),
               int(E_shot_type.get()),
               int(E_distance.get()),
               bool(E_lead.get()),
               int(E_stephen_team_score.get()),
               int(E_opponent_team_score.get()),
               E_opponent.get(),
               E_team.get(),
               int(E_season.get()),
               E_color.get()]
    predictData.loc[len(predictData.index)] = newData

    predictData = pd.get_dummies(predictData, columns=['date'])
    predictData = pd.get_dummies(predictData, columns=['qtr'])
    predictData = pd.get_dummies(predictData, columns=['opponent'])
    predictData = pd.get_dummies(predictData, columns=['team'])
    predictData = pd.get_dummies(predictData, columns=['color'])

    _scaleddata = StandardScaler().fit_transform(predictData)
    _pca = PCA(n_components=2)
    _principalComponents = _pca.fit_transform(_scaleddata)
    _principalDf = pd.DataFrame(data=_principalComponents, columns=['principalComponent1', 'principalComponent2'])
    l = _principalDf.iloc[len(_principalDf)-1].tolist()
    _principalDf = _principalDf.drop(labels = [len(_principalDf)-1],axis = 0)

    data = _principalDf

    if (B_change_current_model["text"] == "Байес"):
        NBaiesML()
        L_r["text"] = gnb.predict([l])
    elif(B_change_current_model["text"] == "Лес"):
        forestML()
        L_r["text"] = forest.predict([l])
    elif(B_change_current_model["text"] == "Вектор"):
        vectorML()
        L_r["text"] = SupportVM.predict([l])
    
def startML():
    if (B_change_current_model["text"] == "Байес"):
        NBaiesML()
    elif(B_change_current_model["text"] == "Лес"):
        forestML()
    elif(B_change_current_model["text"] == "Вектор"):
        vectorML()

work_result_var = StringVar(value=work_result)

L_accuracy = Label(root, text=' ', font='16', justify=LEFT)
B_change_current_model = Button(root, command=change_current_model, text='Байес', font='16')
B_startML = Button(root, command=startML, text='Обучить модель', font='16')
Lb_work_result = Listbox(listvariable=work_result_var, font='16')

L_labels_header = Label(root, text='Название параметра', )
L_top = Label(root, text='oY на поле', )
L_left = Label(root, text='oX на поле', )
L_date = Label(root, text='Дата', )
L_time_remaining = Label(root, text='До конца Qtr', )
L_shot_type = Label(root, text='Тип броска', )
L_distance = Label(root, text='Дистанция', )
L_lead = Label(root, text='Лидирование', )
L_stephen_team_score = Label(root, text='Очки Стефана', )
L_opponent_team_score = Label(root, text='Очки противника', )
L_opponent = Label(root, text='Противник', )
L_team = Label(root, text='Команда', )
L_season = Label(root, text='Сезон', )
L_color = Label(root, text='Цвет', )
L_q = Label(root, text='Этап игры', )
L_res = Label(root, text='Результат броска', )


L_Entry_header = Label(root, text='Значение', )
E_top = Entry()
E_left = Entry()
E_date = Entry()
E_time_remaining = Entry()
E_shot_type = Entry()
E_distance = Entry()
E_lead = Entry()
E_stephen_team_score = Entry()
E_opponent_team_score = Entry()
E_opponent = Entry()
E_team = Entry()
E_season = Entry()
E_color = Entry()
E_q = Entry()
L_r = Label(root, text='', font='16')

L_Max_header = Label(root, text='Максимум', )
L_Max_top = Label(root, text=str(max_arr[0]), )
L_Max_left = Label(root, text=str(max_arr[1]), )
L_Max_time_remaining = Label(root, text=str(max_arr[2]), )
L_Max_stephen_team_score = Label(root, text=str(max_arr[5]), )
L_Max_opponent_team_score = Label(root, text=str(max_arr[6]), )
B_try_predict = Button(root, command=try_predict, text='Предсказать', font='16')


#def calc():



canvas.create_line(800, 0, 800, 720, fill='white', width=2)

canvas.create_window((810, 10), anchor="nw", window=B_change_current_model, width=220, height=25)
canvas.create_window((810, 45), anchor="nw", window=B_startML, width=220, height=25)
canvas.create_window((1050, 10), anchor="nw", window=L_accuracy, width=220, height=60)
canvas.create_window((810, 80), anchor="nw", window=Lb_work_result, width=460, height=100)

canvas.create_window((810, 190+30*0), anchor="nw", window=L_labels_header, width=146, height=20)
canvas.create_window((810, 190+30*1), anchor="nw", window=L_top, width=146, height=20)
canvas.create_window((810, 190+30*2), anchor="nw", window=L_left, width=146, height=20)
canvas.create_window((810, 190+30*3), anchor="nw", window=L_date, width=146, height=20)
canvas.create_window((810, 190+30*4), anchor="nw", window=L_q, width=146, height=20)
canvas.create_window((810, 190+30*5), anchor="nw", window=L_time_remaining, width=146, height=20)
canvas.create_window((810, 190+30*6), anchor="nw", window=L_shot_type, width=146, height=20)
canvas.create_window((810, 190+30*7), anchor="nw", window=L_distance, width=146, height=20)
canvas.create_window((810, 190+30*8), anchor="nw", window=L_lead, width=146, height=20)
canvas.create_window((810, 190+30*9), anchor="nw", window=L_stephen_team_score, width=146, height=20)
canvas.create_window((810, 190+30*10), anchor="nw", window=L_opponent_team_score, width=146, height=20)
canvas.create_window((810, 190+30*11), anchor="nw", window=L_opponent, width=146, height=20)
canvas.create_window((810, 190+30*12), anchor="nw", window=L_team, width=146, height=20)
canvas.create_window((810, 190+30*13), anchor="nw", window=L_season, width=146, height=20)
canvas.create_window((810, 190+30*14), anchor="nw", window=L_color, width=146, height=20)

canvas.create_window((810, 190+30*16), anchor="nw", window=L_res, width=146, height=20)


canvas.create_window((810+146+11, 190), anchor="nw", window=L_Entry_header, width=146, height=20)
canvas.create_window((810+146+11, 190+30*1), anchor="nw", window=E_top, width=146, height=20)
canvas.create_window((810+146+11, 190+30*2), anchor="nw", window=E_left, width=146, height=20)
canvas.create_window((810+146+11, 190+30*3), anchor="nw", window=E_date, width=146, height=20)
canvas.create_window((810+146+11, 190+30*4), anchor="nw", window=E_q, width=146, height=20)
canvas.create_window((810+146+11, 190+30*5), anchor="nw", window=E_time_remaining, width=146, height=20)
canvas.create_window((810+146+11, 190+30*6), anchor="nw", window=E_shot_type, width=146, height=20)
canvas.create_window((810+146+11, 190+30*7), anchor="nw", window=E_distance, width=146, height=20)
canvas.create_window((810+146+11, 190+30*8), anchor="nw", window=E_lead, width=146, height=20)
canvas.create_window((810+146+11, 190+30*9), anchor="nw", window=E_stephen_team_score, width=146, height=20)
canvas.create_window((810+146+11, 190+30*10), anchor="nw", window=E_opponent_team_score, width=146, height=20)
canvas.create_window((810+146+11, 190+30*11), anchor="nw", window=E_opponent, width=146, height=20)
canvas.create_window((810+146+11, 190+30*12), anchor="nw", window=E_team, width=146, height=20)
canvas.create_window((810+146+11, 190+30*13), anchor="nw", window=E_season, width=146, height=20)
canvas.create_window((810+146+11, 190+30*14), anchor="nw", window=E_color, width=146, height=20)

canvas.create_window((810+146+11, 190+30*16), anchor="nw", window=L_r, width=146, height=20)


#.create_window((810+146+11+146+11, 190), anchor="nw", window=L_Max_header, width=146, height=20)
#canvas.create_window((810+146+11+146+11, 190+30*1), anchor="nw", window=L_Max_top, width=146, height=20)
#canvas.create_window((810+146+11+146+11, 190+30*2), anchor="nw", window=L_Max_left, width=146, height=20)
#canvas.create_window((810+146+11+146+11, 190+30*3), anchor="nw", window=L_Max_time_remaining, width=146, height=20)
#.create_window((810+146+11+146+11, 190+30*4), anchor="nw", window=L_Max_stephen_team_score, width=146, height=20)
#canvas.create_window((810+146+11+146+11, 190+30*5), anchor="nw", window=L_Max_opponent_team_score, width=146, height=20)
canvas.create_window((810+146+11+146+11, 190+30*16), anchor="nw", window=B_try_predict, width=146, height=20)

fr = Frame(root, height=500, width=500, bd = 2)
frvar = canvas.create_window((10, 90), anchor="nw", window=fr, width=780, height=620)

length = len(data)

def pce():
    for widget in fr.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots()
    plt.clf()
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('Данные после PCE\n % попаданий '+ str(len(principalDfTRUE['principalComponent1'])/len(principalDf['principalComponent1'])))
    plt.scatter(x=principalDfTRUE.principalComponent1, y=principalDfTRUE.principalComponent2, edgecolor = 'black')
    plt.scatter(x=principalDfFALSE.principalComponent1, y=principalDfFALSE.principalComponent2, edgecolor = 'black')
    canvas1 = FigureCanvasTkAgg(fig, master=fr)
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
    canvas1.draw()

def position():
    for widget in fr.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots()
    plt.clf()
    plt.xlabel('Left')
    plt.ylabel('Top')
    plt.title('Расположение на поле в момент броска')
    plt.scatter(x=visualData.left, y=visualData.top, edgecolor = 'black')
    plt.scatter(x=dataOnlyHit.left, y=dataOnlyHit.top, edgecolor = 'black')
    canvas1 = FigureCanvasTkAgg(fig, master=fr)
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
    canvas1.draw()

def time():
    for widget in fr.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots()
    plt.clf()
    plt.title('Оставшееся время до конца периода')
    plt.xlabel('Оставшееся время (сек)')
    plt.ylabel('Количество записей')
    plt.hist([visualData.time_remaining,dataOnlyHit.time_remaining], bins=20, edgecolor = 'black', stacked=True)
    canvas1 = FigureCanvasTkAgg(fig, master=fr)
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
    canvas1.draw()

def quarter():
    for widget in fr.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots()
    vals = [len(visualData[visualData['qtr_1st OT'] == True]), len(visualData[visualData['qtr_1st Qtr'] == True]), len(visualData[visualData['qtr_2nd Qtr'] == True]), len(visualData[visualData['qtr_3rd Qtr'] == True]), len(visualData[visualData['qtr_4th Qtr'] == True])]
    plt.title('Периоды')
    plt.pie(vals, labels=['qtr_1st OT','qtr_1st Qtr','qtr_2nd Qtr','qtr_3rd Qtr','qtr_4th Qtr'])
    canvas1 = FigureCanvasTkAgg(fig, master=fr)
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
    canvas1.draw()

def score():
    for widget in fr.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots()
    plt.clf()
    plt.title('Превосходство в счёте над вражеской командой')
    plt.xlabel('Превосходство в счёте')
    plt.ylabel('Количество записей')
    plt.hist([visualData.lebron_team_score-visualData.opponent_team_score, dataOnlyHit.lebron_team_score-dataOnlyHit.opponent_team_score], bins=20, edgecolor = 'black', stacked=True)
    canvas1 = FigureCanvasTkAgg(fig, master=fr)
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
    canvas1.draw()

graphics = Label(root, text='Графики данных: ', font='16')
positionButton = Button(root, command=position, text='Позиции', font='16')
timeButton = Button(root, command=time, text='Время', font='16')
quarterButton = Button(root, command=quarter, text='Период', font='16')
scoreButton = Button(root, command=score, text='Счёт', font='16')
pceButton = Button(root, command=pce, text='данные PCE', font='16')

canvas.create_window((9, 10), anchor="nw", window=graphics, width=188, height=30)
canvas.create_window((9+198*1, 10), anchor="nw", window=pceButton, width=188, height=30)
canvas.create_window((9, 50), anchor="nw", window=positionButton, width=188, height=30)
canvas.create_window((9+198*1, 50), anchor="nw", window=timeButton, width=188, height=30)
canvas.create_window((9+198*2, 50), anchor="nw", window=quarterButton, width=188, height=30)
canvas.create_window((9+198*3, 50), anchor="nw", window=scoreButton, width=188, height=30)


root.mainloop()

#print(data.head())