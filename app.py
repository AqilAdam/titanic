from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)

W=None
b=None


def sig(x):
  return 1/(1+np.exp(-x))


def predict(X,W,b):
  print(X)
  return sig(np.dot(X,W)+b)


with open("models/model_params.pkl","rb") as f:
    model_params=pickle.load(f)
    W=model_params["Weight"]
    b=model_params["Bias"]

with open("models/scaler.pkl","rb") as f:
    scaler=pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction=None
    if request.method == 'POST':

        pclass = float (request.form.get('pclass'))
        age = float(request.form.get('age'))
        sbsp = float(request.form.get('sbsp'))
        parch = float(request.form.get('parch'))
        fare = float(request.form.get('fare'))
        sex = request.form.get('sex')
        port = request.form.get('port')

        if(sex=='Male'):
            Sex_male=1
        else:
            Sex_male=0

        if(port=="C"):
            Embarked_Q=0	
            Embarked_S=0
        elif(port=="Q"):
            Embarked_Q=1	
            Embarked_S=0
        else:
            Embarked_Q=0	
            Embarked_S=1            
            

        print("Weights are=",W)
        print("Bias is:",b)
        print("Scaler is",scaler,"min is=",scaler.min_)

        dict=[{"Pclass":pclass,"Age":age,"SibSp":sbsp,"Parch":parch,"Fare":fare,"Sex_male":Sex_male,"Embarked_Q":Embarked_Q,"Embarked_S":Embarked_S}]
        df=pd.DataFrame(dict)
        print(df)
        df[['Age','Fare']]=scaler.transform(df[['Age','Fare']])
        print(df)
        for i in df.columns:
            print(df[i], "+ ")


        print("PREDICTINGGG PARTTTT")


        X=df.to_numpy()
        print(X)
        
        pred=predict(X,W,b)
        print("The value predicted is= ",pred)
        prediction = "Survived" if pred[0] >= 0.5 else "Did not Survive"
    return render_template('index.html',prediction = prediction)
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

#['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
       #'Embarked_S']