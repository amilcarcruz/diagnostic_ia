# importamos las librerías que necesitamos
from sklearn.datasets import load_diabetes  # datos de diabetes
from sklearn.tree import DecisionTreeClassifier  # árbol de decisión para clasificación


def load_data_diabetes():
    return load_diabetes()


def get_description_diabetes_data():
    diabetes_data = load_data_diabetes()
    return diabetes_data.DESCR


def get_4_row():
    return load_diabetes.data[48:52, :]


if __name__ == "__main__":
    # print(get_description_diabetes_data())
    print(get_4_row())
