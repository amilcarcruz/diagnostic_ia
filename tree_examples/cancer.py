# importamos las librerías que necesitamos
from sklearn.datasets import load_breast_cancer  # datos de diabetes
from sklearn.tree import DecisionTreeClassifier  # árbol de decisión para clasificación


def load_data_cancer():
    return load_breast_cancer()


def get_description_cancer_data():
    cancer_data = load_data_cancer()
    return cancer_data.DESCR


if __name__ == "__main__":
    print(get_description_cancer_data())
