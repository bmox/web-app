import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle

orig_url = 'https://drive.google.com/file/d/1ZxaThxH4fqtZQxxdgVQcD2dUDULYXLxo/view?usp=sharing'
file_id = orig_url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
credit = pd.read_csv(csv_raw)

credit.rename(columns={"expenditure":"expe","dependents":"depen","majorcards":"major"}, inplace = True)

cleanup_nums = {"card":{"yes": 1, "no": 0}}
credit.replace(cleanup_nums, inplace=True)

x_o=credit[['expe', 'share', 'reports', 'age', 'income']]
y_o=credit["card"]

scaling = MinMaxScaler(feature_range=(0, 1))
new_x = scaling.fit_transform(x_o)
x_train = new_x[:1000]
x_test = new_x[1000:1319]
y_train = y_o[:1000]
y_test = y_o[1000:1319]

dt_model = DecisionTreeClassifier()
dt_model=dt_model.fit(x_train,y_train)
pickle.dump(dt_model,open('dt_model.pkl','wb'))


