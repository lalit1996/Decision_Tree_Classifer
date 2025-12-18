import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.metrics import accuracy_score,f1_score

data = {
    'weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy',
                'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast',
                'Overcast', 'Rainy'],

    'temperature': [30, 32, 25, 22, 20, 18, 26, 28, 24, 21, 27, 23, 29, 19],

    'humidity': [85, 90, 78, 80, 70, 65, 72, 95, 70, 75, 80, 68, 75, 85],

    'wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak',
             'Strong', 'Strong'],

    'play_football': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                      'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
                      'Yes', 'No']
}

df = pd.DataFrame(data)

independent = df.drop(columns=['play_football'],axis=1)
dependent = df['play_football']

print(independent.columns)

inde_coltrans = ColumnTransformer(
    transformers=[
        ('cat1',OneHotEncoder(sparse_output=False),['weather']),
        ('cat2',OrdinalEncoder(),['wind']),
        ('Num','passthrough',['temperature','humidity'])
    ]
)

Ordinal = OrdinalEncoder()

x_train, x_test, y_train, y_test = train_test_split(independent,dependent,test_size=0.2)

x_train = inde_coltrans.fit_transform(x_train)
y_train = Ordinal.fit_transform(pd.DataFrame(y_train))
x_test = inde_coltrans.transform(x_test)
y_test = Ordinal.transform(pd.DataFrame(y_test))


print(x_train)
print(y_train)
print(x_test)

Dt = DecisionTreeClassifier()
Dt.fit(x_train,y_train)
y_predict = Dt.predict(x_test)


print(y_predict)
print(y_test)