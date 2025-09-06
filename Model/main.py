import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully!")

    # test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:' , accuracy_score(y_test, y_pred))
    print("Classification report: \n" , classification_report(y_test,y_pred))

    return model, scaler




def get_clean_data():
    print("Fetching data...")
    try:
        data = pd.read_csv("D:/Streamlit_APP_Cancer/Data/data.csv")
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: data.csv file not found!")
        return None

    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    print("Data cleaned successfully!")
    return data

def main():
    print("Starting the script...")
    data = get_clean_data()

    if data is None:
        print("Exiting due to missing data file.")
        return

    model, scaler = create_model(data)
    with open('Model/model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('Model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    print("Script finished successfully!")


if __name__ == '__main__':
    main()
