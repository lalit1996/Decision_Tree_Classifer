import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score,f1_score,r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

Dataframe = pd.read_csv(r"C:/Users/dell/OneDrive/Desktop/Machine Learning Model/Decision_Tree_Classifer/DecisionTree/toyota.csv", delimiter=',', encoding='latin1')

print(Dataframe.head(10).to_string())


len(Dataframe)

Target = Dataframe['price']
Independent = Dataframe.drop(columns=['model','price'],axis=1)
print(Independent.columns)

columntrans = ColumnTransformer(
    transformers=[
        ('Cat',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first'),['fuelType','transmission']),
        ('Num','passthrough',['year','mileage','tax','mpg','engineSize'])
    ]
)


x_train, x_test, y_train, y_test = train_test_split(Independent,Target,test_size=0.2)

x_train = columntrans.fit_transform(x_train)
x_test = columntrans.transform(x_test)


Dectree = DecisionTreeRegressor()
ccp_alphas = Dectree.cost_complexity_pruning_path(x_train,y_train)

Final_Train_Testing_Score = []
Final_test_testing_score = []
ccp_val = list(ccp_alphas['ccp_alphas'])




StFd = StratifiedKFold(n_splits=5)
for i in ccp_alphas['ccp_alphas']:
    train_testing_Score = []
    test_testing_score = []
    for train_loc,test_loc in StFd.split(x_train,y_train):
        x_train_S = x_train[train_loc]
        y_train_S = y_train.iloc[train_loc]
        x_test_S = x_train[test_loc]
        y_test_S = y_train.iloc[test_loc]
        Dectree = DecisionTreeRegressor(ccp_alpha=i)
        Dectree.fit(x_train_S,y_train_S)
        predict_trainY = Dectree.predict(x_train_S)
        predict_TestY = Dectree.predict(x_test_S)
        train_testing_Score.append(r2_score(y_train_S,predict_trainY))
        test_testing_score.append(r2_score(y_test_S,predict_TestY))

    Final_Train_Testing_Score.append(np.mean(train_testing_Score))
    Final_test_testing_score.append((np.mean(test_testing_score)))

best_alpha = ccp_val[np.argmax(Final_test_testing_score)]


Dectree = DecisionTreeRegressor(ccp_alpha=best_alpha)
Dectree.fit(x_train,y_train)
predicted_y = Dectree.predict(x_test)

print(r2_score(y_test,predicted_y))
print(mean_absolute_error(y_test,predicted_y))
print(mean_squared_error(y_test,predicted_y))


# GSearch = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid={"ccp_alpha":ccp_val},cv=StratifiedKFold(n_splits=5))

# GSearch.fit(x_train,y_train)
# print(GSearch.best_params_)
# print(accuracy_score(y_test,GSearch.predict(x_test)))


# ---------------------------------------------------- Solve through Linear Regression -----------------------------------------------------
#
# columntrans = ColumnTransformer(
#         transformers=[
#             ('Cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), ['fuelType', 'transmission']),
#             ('Num', 'passthrough', ['year', 'mileage', 'tax', 'mpg', 'engineSize']),
#             ('Standscale', StandardScaler(),['year', 'mileage', 'tax', 'mpg', 'engineSize'])
#         ]
#     )
#
# Dpipe = Pipeline(
#     [
#         ('Coltrans',columntrans),
#         ('Train',LinearRegression())
#     ]
# )
#
# TransformedTargetRegressor = TransformedTargetRegressor(
#     regressor=Dpipe,
#     transformer=StandardScaler()
# )
#
#
# stratefieldfold = StratifiedKFold(n_splits=5)
#
# for train_loc, test_loc in stratefieldfold.split(x_train,y_train):
#     k_fold_x_train = x_train.iloc[train_loc]
#     k_fold_y_train = y_train.iloc[train_loc]
#     k_fold_x_test = x_train.iloc[test_loc]
#     k_fold_y_test = y_train.iloc[test_loc]
#     Linear_model_training = TransformedTargetRegressor.fit(k_fold_x_train,k_fold_y_train)
#     k_fold_y_predict = Linear_model_training.predict(k_fold_x_test)
#     print(r2_score(k_fold_y_test,k_fold_y_predict))
#     print(mean_squared_error(k_fold_y_test,k_fold_y_predict))
#     print(mean_absolute_error(k_fold_y_test,k_fold_y_predict))
    # print(k_fold_y_predict)




