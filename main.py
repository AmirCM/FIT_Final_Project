import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape


def count_result(ser: pd.Series):
    return ser.unique(), ser.value_counts()


df = pd.read_excel('datasets/FIT.xlsx', engine='openpyxl')
df = df.loc[:, :'kyw']
df.rename(columns={'g': 'name', 'a Type': 'type', 'a Time': 'date', 'j Tiltle': 'job title', 'rmtWrk': 'remote',
                   'kyw': 'key words'}, inplace=True)

cities = {i: city for i, city in enumerate(df['city'].unique()) if city is not np.nan}

cities_cleaned = {}
for i, city in cities.items():
    spell = city.split(' ')
    if len(spell) == 1:
        cities_cleaned[spell[0]] = 0
    elif len(spell[0]) < 1 or len(spell[1]) < 1:
        if len(spell[0]) > 1:
            cities_cleaned[spell[0]] = 0
        else:
            cities_cleaned[spell[1]] = 0
    elif len(spell) == 2:
        cities_cleaned[city] = 0

for city in cities_cleaned:
    cities_cleaned[city] = [c for c in cities.values() if city in c and abs(len(c) - len(city)) < 2]

cities_df = {}
number_of_advertise = np.nan

for k, v in cities_cleaned.items():
    main_df = []
    for c in v:
        temp_df = df[df['city'] == c]
        main_df.append(temp_df)

    cities_df[k] = pd.concat(main_df)
    if number_of_advertise is np.nan:
        number_of_advertise = np.array([cities_df[k].shape[0]])
    else:
        number_of_advertise = np.append(number_of_advertise, [cities_df[k].shape[0]])

labels = cities_df.keys()
persian_labels = [get_display(reshape(label)) for label in labels]

fig, ax = plt.subplots()
ax.bar(persian_labels, number_of_advertise, color='g')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(number_of_advertise):
    plt.text(xlocs[i] - 0.25, v + 10, str(v))

plt.xlabel(f'Cities: {len(cities_cleaned)}')
plt.ylabel('Number of jobs')
fig.autofmt_xdate()
plt.title('This chart illustrate the number of job request per city')
plt.show()
