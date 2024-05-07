# Ex.No: 13 Learning – Use Supervised Learning  

### DATE: 30/04/2024             
 
### REGISTER NUMBER : 212221040078

## AIM: 
 
To write a program to train the classifier for Diabetes Prediction.

##  Algorithm:

1.Start the program.

2.Import required Python libraries, including NumPy, Pandas, Google Colab, Gradio, and various scikit-learn modules.

3.Mount Google Drive using Google Colab's 'drive.mount()' method to access the data file located in Google Drive.

4.Install the Gradio library using 'pip install gradio'.

5.Load the diabetes dataset from a CSV file ('diabetes.csv') using Pandas.

6.Separate the target variable ('Outcome') from the input features and Scale the input features using the StandardScaler from scikit-learn.

7.Create a multi-layer perceptron (MLP) classifier model using scikit-learn's 'MLPClassifier'.

8.Train the model using the training data (x_train and y_train).

9.Define a function named 'diabetes' that takes input parameters for various features and Use the trained machine learning model to predict the outcome based on the input features.

10.Create a Gradio interface using 'gr.Interface' and Specify the function to be used to make predictions based on user inputs.

11.Launch the Gradio web application, enabling sharing, to allow users to input their data and get predictions regarding diabetes risk.

12.Stop the program.

### Program:
```
import numpy as np
import pandas as pd

pip install gradio
pip install typing-extensions --upgrade
import gradio as gr
```
```
data = pd.read_csv('diabetes.csv')
data.head()
```
![image](https://github.com/Anbuselvan04/AI_Lab_2023-24/assets/119410896/5769c7ba-260a-4d47-afec-c633efa6495e)

```
print(data.columns)
```
![image](https://github.com/Anbuselvan04/AI_Lab_2023-24/assets/119410896/f88d9a32-d0e2-4ee1-bd4b-9470debd021d)

```
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
```
![image](https://github.com/Anbuselvan04/AI_Lab_2023-24/assets/119410896/ff7ae4f8-6619-4d57-b3e2-c8fa3d67cb84)

```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y)

#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#instatiate model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))
```
![image](https://github.com/Anbuselvan04/AI_Lab_2023-24/assets/119410896/60bcd502-3df1-411b-8063-29f5352cede5)

```
#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```
![image](https://github.com/Anbuselvan04/AI_Lab_2023-24/assets/119410896/c097c01b-6e4c-4643-8a22-965fa22caee7)

### Result:
Thus the system was trained successfully and the prediction was carried out.
