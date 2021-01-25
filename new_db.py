import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

frame_lan = {'python': ['flask', 'django', 'cherrypy', 'falcon'],
             'javascript': ['angular', 'jquery', 'java-script', 'vue', 'react'],
             'html': ['bootstrap', 'html5', 'foundation', 'skeleton'],
             'css': ['bootstrap', 'html5', 'foundation', 'skeleton'],
             'php': ['laravel', 'cakephp', 'symphony', 'zend'],
             'c#': ['asp', '.net', 'asp.net', 'c-#', '#c'],
             'c++': ['cpp'],
             'sql': ['sqlserver', 'sqlite']}


def check_stem(words: list):
    i_words = set()
    for w in words:
        find = False
        for k, v in frame_lan.items():
            if k in w:
                i_words.add(k)
                find = True
                break
            for f in v:
                if f in w:
                    i_words.add(k)
                    find = True

        if not find:
            i_words.add(w)
    return list(i_words)


def find_common(entries):
    all_words = []
    for text in entries:
        if isinstance(text, str):
            key_w = [word.lower() for word in text.split(' ') if
                     re.match(r'^[a-zA-Z+#]+[a-zA-Z0-9-_+@#%&*|]+$|^c$', word)]
            all_words += check_stem(key_w)

    all_words = nltk.FreqDist(all_words)
    return all_words.most_common()


class City:
    def __init__(self, df: pd.DataFrame, name):
        self.df = df
        self.name = name
        self.number_of_advertise = self.df.shape[0]

    def trend(self):
        entries = self.df['key words'].values.tolist()
        # print(entries)
        return find_common(entries)


df = pd.read_excel('datasets/dataset.xlsx', engine='openpyxl')
df = df.loc[:, :'kyw']
df.rename(columns={'g': 'name', 'a Type': 'type', 'a Time': 'date', 'j Tiltle': 'job title', 'rmtWrk': 'remote',
                   'kyw': 'key words'}, inplace=True)

name_of_cities = [city for i, city in enumerate(df['city'].unique()) if city is not np.nan]

cities = []

for c in name_of_cities:
    cities.append(City(df[df['city'] == c], c))

fig = plt.figure()
labels = []
popularity = []
programming_language = {}

"""
with open('keys.txt', 'w') as f:
    for i, t in enumerate(cities[0].trend()):
        try:
            f.write(f'{t}\n')
            lang = ''
            for k, v in frame_lan.items():
                for frame_work in v:
                    if frame_work in t[0]:
                        lang = k
                        break
            if lang:
                if lang in languages:
                    languages[lang] += t[1]
                else:
                    languages[lang] = t[1]
            else:
                if languages[k]:
                    languages[k] += t[1]
                else:
                    languages[k] = t[1]

        except:
            print(f'{t[0]}: {t[1]}')

    languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    print(languages)

"""
for j, city in enumerate(cities):

    print(city.name)
    labels = []
    popularity = []
    for i, w in enumerate(city.trend()):
        if city.number_of_advertise * 0.05 > w[1] and city.number_of_advertise != 1:
            break
        labels += [w[0]]
        popularity += [w[1]]

    if j % 2 == 0 and j != 0:
        fig = plt.figure()
    axs = fig.add_subplot(int(f'21{j % 2 + 1}'))
    axs.set_title(f'Trends in {city.name}')
    axs.bar(labels, popularity)
    plt.setp(axs.get_xticklabels(), rotation=30, horizontalalignment='right')

plt.show()

"""ads = [city.number_of_advertise for city in cities]
fig, ax = plt.subplots()
ax.bar(name_of_cities, ads, color='g')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(ads):
    plt.text(xlocs[i] - 0.25, v + 10, str(v))

plt.xlabel(f'Cities: {len(cities)}')
plt.ylabel('Number of jobs')
fig.autofmt_xdate()
plt.title('This chart illustrate the number of job request per city')
plt.show()"""
