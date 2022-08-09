import pandas as pd

DEMO_URL = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
adult_path = "datasets/adult.csv"
titanic_path = "datasets/titanic.csv"
heart1_path = "datasets/heart1.csv"
heart2_path = "datasets/heart2.csv"
lung_path = "datasets/lung.csv"
breast_path = "datasets/breast.csv"
diabetes_path = "datasets/diabetes.csv"
ecoli_path = "datasets/ecoli.csv"
yeast6_path = "datasets/yeast6.csv"
yeast0_path = "datasets/yeast.csv"
abalone_path = "datasets/abalone.csv"
flare_path = "datasets/flare.csv"



def load_demo():
    return pd.read_csv(DEMO_URL, compression='gzip')

def load_adult():
    return pd.read_csv(adult_path)

def load_titanic():
    return pd.read_csv(titanic_path)

def load_flare():
    return pd.read_csv(flare_path)

def load_heart1():
    return pd.read_csv(heart1_path)

def load_heart2():
    return pd.read_csv(heart2_path)

def load_lung():
    return pd.read_csv(lung_path)

def load_breast():
    return pd.read_csv(breast_path)


def load_diabetes():
    return pd.read_csv(diabetes_path)

def load_ecoli():
    return pd.read_csv(ecoli_path)

def load_yeast6():
    return pd.read_csv(yeast6_path)

def load_yeast0():
    return pd.read_csv(yeast0_path)

def load_abalone():
    return pd.read_csv(abalone_path)
