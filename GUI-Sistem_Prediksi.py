import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from sklearn.metrics import ConfusionMatrixDisplay

def open_file():
    file_path = filedialog.askopenfilename()
    global data
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.shape)
    
    # Perform data processing and modeling here
    # Create new column 'average_score'
    data['average_score'] = np.int_(data[['math score', 'reading score', 'writing score']].mean(axis=1))

    # Define function to convert average_score to letter grade
    def letter_grade(average_score):
        if average_score >= 90:
            return 'A'
        elif average_score < 90 and average_score >= 80:
            return 'B'
        elif average_score < 80 and average_score >= 70:
            return 'C'
        elif average_score < 70 and average_score >= 60:
            return 'D'
        else:
            return 'E'

    # Apply letter_grade function to create new column 'grades'
    data['grades'] = data.apply(lambda x: letter_grade(x['average_score']), axis=1)

def run_model():
    global data
    # Encode categorical variables using LabelEncoder
    le = LabelEncoder()
    for x in data:
        if data[x].dtypes == 'object':
            data[x] = le.fit_transform(data[x])
    print(utils.multiclass.type_of_target(data[x].astype('int')))
    print(data.head())
    
    # Drop unnecessary columns and split dataset into train and test sets
    # data = data.drop(columns=['average_score'])
    x = data.drop(columns=['grades'])
    y = data['grades']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)

    if algorithm.get() == "naive_bayes":
        # Naive Bayes
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        print(accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

    elif algorithm.get() == "logistic_regression":
        # Logistic Regression
        model = LogisticRegression(random_state=13)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        print(accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

    elif algorithm.get() == "random_forest":
        # Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=13)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        print(accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

    else:
        print("Silakan pilih algoritma")

root = tk.Tk()
root.title("Dataset Upload and Analysis")
root.geometry("300x300")

upload_button = tk.Button(root, text="Upload Dataset", command=open_file)
upload_button.pack(pady=20)

# Create radio button to choose algorithm
algorithm = tk.StringVar()

nb_radio = tk.Radiobutton(root, text="Naive Bayes", variable=algorithm, value="naive_bayes")
rf_radio = tk.Radiobutton(root, text="Random Forest", variable=algorithm, value="random_forest")
lr_radio = tk.Radiobutton(root, text="Logistic Regression", variable=algorithm, value="logistic_regression")

nb_radio.pack()
rf_radio.pack()
lr_radio.pack()

run_button = tk.Button(root, text="Jalankan Algortima", command=run_model)
run_button.pack(pady=20)

root.mainloop()