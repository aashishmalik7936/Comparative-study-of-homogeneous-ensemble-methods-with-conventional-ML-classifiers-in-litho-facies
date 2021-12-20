import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import scipy
from sklearn.metrics import r2_score



data_1 = pd.read_csv('data_set_1.csv')
data_2 = pd.read_csv('data_set_2.csv')
data_3 = pd.read_csv('data_set_3.csv')
data_4 = pd.read_csv('data_set_4.csv')
data=pd.concat([data_1, data_2, data_3, data_4], ignore_index=True)


data=data.drop([' Date Time'], axis=1)

data.loc[(data['Depth']<8887), 'formation']=0
data.loc[(data['Depth']>=8887)&(data['Depth']<9044), 'formation']=1
data.loc[(data['Depth']>=9044)&(data['Depth']<9462), 'formation']=2
data.loc[(data['Depth']>=9462)&(data['Depth']<9590), 'formation']=3
data.loc[(data['Depth']>=9590)&(data['Depth']<11684), 'formation']=4
data.loc[(data['Depth']>=11684)&(data['Depth']<11801), 'formation']=3
data.loc[(data['Depth']>=11801), 'formation']=4

data=data.loc[(data['D-EXPONENT']!='#NUM!') & (data['D-EXPONENT']!='#DIV/0!')]
data['D-EXPONENT']=data['D-EXPONENT'].astype(float)

data=data[(data['Gamma Ray']!=-999.25) & (data['Hook Load']!=-999.25)]
data.info()


X=data.drop(['formation'], axis=1, inplace=False)
Y=data['formation']

depth=data.iloc[:, 0]

train_X=X.iloc[:24819, 1:15]
train_y=Y.iloc[:24819]
test_X=X.iloc[24819:, 1:15]
test_y=Y.iloc[24819:]


encoder = LabelEncoder()
y_train=encoder.fit_transform(train_y.values)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(train_X.values)

X_test=scaler.transform(test_X.values)
y_test=encoder.transform(test_y.values)


# Logistic Regression
val_f1_score=0
test_f1_score=0
test_accuracy_score=0
test_predictions=[]

skf=KFold(n_splits=5, random_state=42)
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, X_val=X_train[list(tr_idx)], X_train[list(val_idx)]
    y_tr, y_val=y_train[list(tr_idx)], y_train[list(val_idx)]

    model=LogisticRegression(random_state=42).fit(X_tr, y_tr)
    val_pred=model.predict(X_val)

    val_f1_score+=f1_score(y_val, val_pred, average='weighted')

    test_pred=model.predict(X_test)
    test_f1_score+=f1_score(y_test, test_pred, average='weighted')
    test_accuracy_score+=accuracy_score(y_test, test_pred)
    test_predictions=test_pred
print('avg val f1_score: ', val_f1_score/5)
print('test f1_score: ', test_f1_score/5)
print('test accuracy_score: ', test_accuracy_score/5)

scipy.stats.pearsonr(test_predictions, y_test)

r2_score(y_test, test_predictions)



# SVM
val_f1_score=0
test_f1_score=0
test_accuracy_score=0
test_predictions=[]

skf=KFold(n_splits=5, random_state=42)
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, X_val=X_train[list(tr_idx)], X_train[list(val_idx)]
    y_tr, y_val=y_train[list(tr_idx)], y_train[list(val_idx)]

    model=SVC(random_state=42).fit(X_tr, y_tr)
    val_pred=model.predict(X_val)

    val_f1_score+=f1_score(y_val, val_pred, average='weighted')

    test_pred=model.predict(X_test)
    test_f1_score+=f1_score(y_test, test_pred, average='weighted')
    test_accuracy_score+=accuracy_score(y_test, test_pred)
    test_predictions=test_pred
    
print('avg val f1_score: ', val_f1_score/5)
print('test f1_score: ', test_f1_score/5)
print('test accuracy_score: ', test_accuracy_score/5)

scipy.stats.pearsonr(test_predictions, y_test)

r2_score(y_test, test_predictions)


# Random Forest
val_f1_score=0
test_f1_score=0
test_accuracy_score=0
test_predictions=[]

skf=KFold(n_splits=5, random_state=42)
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, X_val=X_train[list(tr_idx)], X_train[list(val_idx)]
    y_tr, y_val=y_train[list(tr_idx)], y_train[list(val_idx)]

    model=RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=42, max_depth=6).fit(X_tr, y_tr)
    val_pred=model.predict(X_val)

    val_f1_score+=f1_score(y_val, val_pred, average='weighted')

    test_pred=model.predict(X_test)
    test_f1_score+=f1_score(y_test, test_pred, average='weighted')
    test_accuracy_score+=accuracy_score(y_test, test_pred)
    test_predictions=test_pred

print('avg val f1_score: ', val_f1_score/5)
print('test f1_score: ', test_f1_score/5)
print('test accuracy_score: ', test_accuracy_score/5)

scipy.stats.pearsonr(test_predictions, y_test)

r2_score(y_test, test_predictions)


# Adaboost
val_f1_score=0
test_f1_score=0
test_accuracy_score=0
test_predictions=[]

skf=KFold(n_splits=5, random_state=42)
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, X_val=X_train[list(tr_idx)], X_train[list(val_idx)]
    y_tr, y_val=y_train[list(tr_idx)], y_train[list(val_idx)]

    model=AdaBoostClassifier(n_estimators=30, random_state=42).fit(X_tr, y_tr)
    val_pred=model.predict(X_val)

    val_f1_score+=f1_score(y_val, val_pred, average='weighted')

    test_pred=model.predict(X_test)
    test_f1_score+=f1_score(y_test, test_pred, average='weighted')
    test_accuracy_score+=accuracy_score(y_test, test_pred)
    test_predictions=test_pred

print('avg val f1_score: ', val_f1_score/5)
print('test f1_score: ', test_f1_score/5)
print('test accuracy_score: ', test_accuracy_score/5)

scipy.stats.pearsonr(test_predictions, y_test)

r2_score(y_test, test_predictions)


# XGboost
val_f1_score=0
test_f1_score=0  #subsample=0.3 and max_depth=6 and n_estimators=500
test_accuracy_score=0
test_predictions=[]

skf=KFold(n_splits=5, random_state=42)
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, X_val=X_train[list(tr_idx)], X_train[list(val_idx)]
    y_tr, y_val=y_train[list(tr_idx)], y_train[list(val_idx)]

    model=XGBClassifier(subsample=0.3, tree_method='exact', min_child_weight=0, max_delta_step=0.3, booster='gbtree',n_estimators=500, max_depth=6, objective='multi:softmax', random_state=42).fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mlogloss', verbose=True)
    val_pred=model.predict(X_val)

    val_f1_score+=f1_score(y_val, val_pred, average='weighted')

    test_pred=model.predict(X_test)
    test_f1_score+=f1_score(y_test, test_pred, average='weighted')
    test_accuracy_score+=accuracy_score(y_test, test_pred)
    test_predictions=test_pred

print('avg val f1_score: ', val_f1_score/5)
print('test f1_score: ', test_f1_score/5)
print('test accuracy_score: ', test_accuracy_score/5)

scipy.stats.pearsonr(test_predictions, y_test)

r2_score(y_test, test_predictions)

# Predicting using XGboost model
X_=X.iloc[:, 1:].values
X_scaled=scaler.transform(X_)
y_encoded=encoder.transform(Y.values)
predictions=model.predict(X_scaled)

# Ploting the line plot between real and predicted values
final=np.column_stack(((depth, predictions, y_encoded)))
final=final[final[:, 0].argsort()[::-1][:]]
final_df=pd.DataFrame(final, columns=['Depth', 'predicted', 'actual'])


df = final_df.melt('Depth', var_name='Legend',  value_name='Formation code')
ax = sns.lineplot(x="Depth", y="Formation code", hue='Legend', data=df, ci=None, estimator=np.median)

x, y_actual=ax.lines[1].get_data()
x, y_pred=ax.lines[0].get_data()

a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims, facecolor='white', dpi=500)
plt.rcParams['font.family'] = "Times New Roman"
legend_properties = {'weight':'bold'}
ax.grid(False)
ax.set_facecolor('white')
plt.plot(y_pred, x, label = "Predicted", color='r')
plt.plot(y_actual, x, label = "Actual", color='b')
ax.invert_yaxis()
ax.xaxis.tick_top()   
ax.xaxis.set_label_position('top') 
plt.ylabel("Depth", fontsize=14, fontweight="bold")
plt.xlabel("Formation code", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.legend(prop=legend_properties)
plt.savefig('final_litho_new.jpg')
