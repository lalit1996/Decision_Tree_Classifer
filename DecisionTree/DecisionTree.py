import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

Dataframe = pd.read_csv("/Decision_Tree_Classifer/DecisionTree/toyota.csv", delimiter=',', encoding='latin1')


len(Dataframe)

Target = Dataframe['transmission']
Independent = Dataframe.drop(columns=['model','transmission'],axis=1)
print(Independent.columns)

columntrans = ColumnTransformer(
    transformers=[
        ('Cat',OneHotEncoder(sparse_output=False),['fuelType']),
        ('Num','passthrough',['year','mileage','tax','mpg','engineSize','price'])
    ]
)

ordinal = OrdinalEncoder()
x_train, x_test, y_train, y_test = train_test_split(Independent,Target,test_size=0.2)

x_train = columntrans.fit_transform(x_train)
x_test = columntrans.transform(x_test)
y_train = ordinal.fit_transform(pd.DataFrame(y_train))
y_test = ordinal.transform(pd.DataFrame(y_test))


ccp_alphas = Dectree.cost_complexity_pruning_path(x_train,y_train)

Final_Train_Testing_Score = []

Final_test_testing_score = []
ccp_val = list(ccp_alphas['ccp_alphas'])
GSearch = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid={"ccp_alpha":ccp_val},cv=StratifiedKFold(n_splits=5))

GSearch.fit(x_train,y_train)
print(GSearch.best_params_)
print(accuracy_score(y_test,GSearch.predict(x_test)))

StFd = StratifiedKFold(n_splits=5)
for i in ccp_val:
    train_testing_Score = []
    test_testing_score = []
    for train_loc,test_loc in StFd.split(x_train,y_train):
        x_train_S = x_train[train_loc]
        y_train_S = y_train[train_loc]
        x_test_S = x_train[test_loc]
        y_test_S = y_train[test_loc]
        Dectree = DecisionTreeClassifier(ccp_alpha=i)
        Dectree.fit(x_train_S,y_train_S)
        predict_trainY = Dectree.predict(x_train_S)
        predict_TestY = Dectree.predict(x_test_S)
        train_testing_Score.append(accuracy_score(y_train_S,predict_trainY))
        test_testing_score.append(accuracy_score(y_test_S,predict_TestY))

    Final_Train_Testing_Score.append(np.mean(train_testing_Score))
    Final_test_testing_score.append((np.mean(test_testing_score)))

print(Final_Train_Testing_Score)
print(Final_test_testing_score)

plt.plot(ccp_val,Final_Train_Testing_Score)
plt.plot(ccp_val,Final_test_testing_score)
plt.legend()
plt.show()




