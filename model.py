import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle


statement_account = pd.read_csv('bank.csv') 
for column in statement_account.columns:
    statement_account[column].fillna(0, inplace=True)
statement_account.dropna()
statement_account.drop_duplicates()
last_30_transactions = statement_account.head(30)
last_30_transactions['WITHDRAWAL AMT'] = last_30_transactions['WITHDRAWAL AMT'].apply(lambda x: ''.join(filter(str.isdigit, str(x))) if pd.notnull(x) else x)
last_30_transactions['WITHDRAWAL AMT'] = last_30_transactions['WITHDRAWAL AMT'].astype(int, errors= 'ignore')
average_transactions = last_30_transactions['WITHDRAWAL AMT'].mean()
anomaly_threshold = 1.5
new_transaction = 10000

def train_model():
    X = np.array([[average] for average in last_30_transactions['WITHDRAWAL AMT']])
    y = np.array([(transaction > (anomaly_threshold * average_transactions)) for transaction in last_30_transactions['WITHDRAWAL AMT']])
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.2)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    is_anomaly = model.predict([[new_transaction]])
    if is_anomaly: 
        return f"The transaction ({new_transaction}) is predicted to be an anomaly."
    else:
        return f"The transaction ({new_transaction}) is not predicted to be an anomaly."

    pkl_filename = "./hthub_model.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    train_model()

