import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_result(ser: pd.Series):
    return ser.unique(), ser.value_counts()


df = pd.read_excel('datasets/FIT.xlsx', engine='openpyxl')
df = df.loc[:, :'kyw']
df.rename(columns={'g': 'name', 'a Type': 'type', 'a Time': 'date', 'j Tiltle': 'job title', 'rmtWrk': 'remote',
                   'kyw': 'key words'}, inplace=True)

cities = df['city'].unique()
print(cities)
