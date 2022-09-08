import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def read_file_data_diabetes():
    df = pd.read_csv('/Users/orquidea/Projects/ia/tree_examples/diabetes_dataset.csv')
    return df.head()


def get_variables():
    df = read_file_data_diabetes()
    # Feature variables
    x = df.drop(['Outcome'], axis=1)
    # Target variable
    y = df.Outcome
    return x, y


def build_tree_decision():
    x, y = get_variables()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Create Decision Tree Classifer object
    model = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    model = model.fit(x_train, y_train)

    # Train Decision Tree Classifer
    y_pred = model.predict(x_test)

    return y_pred


if __name__ == "__main__":
    print(build_tree_decision())
