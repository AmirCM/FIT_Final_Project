import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

frame_lan = {'python': ['flask', 'django', 'cherrypy', 'falcon'],
             'javascript': ['angular', 'jquery', 'vue', 'react'],
             'html': ['bootstrap', 'html5', 'foundation', 'skeleton'],
             'css': ['bootstrap', 'html5', 'foundation', 'skeleton'],
             'php': ['laravel', 'cakephp', 'symphony', 'zend'],
             'c#': ['asp', '.net', 'asp.net', 'c-#', '#c'],
             'c++': ['cpp'],
             'sql': ['sqlserver', 'sqlite']}

languages = [lan for lan in frame_lan]
languages += ['java', 'swift', 'c']


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
    frame_works = []
    for text in entries:
        if isinstance(text, str):
            key_w = [word.lower() for word in text.split(' ') if
                     re.match(r'^[a-zA-Z+#]+[a-zA-Z0-9-_+@#%&*|]+$|^c$', word)]
            frame_works += key_w
            key_w = check_stem(key_w)
            all_words += key_w

    all_words = nltk.FreqDist(all_words)
    frame_works = nltk.FreqDist(frame_works)
    return all_words.most_common(), frame_works.most_common()


def draw(x, y, title):
    fig = plt.figure()
    labels = []
    popularity = []
    for i, w in enumerate(x):
        if i > 10:
            break
        labels += [w[0] + f': {w[1]}']
        popularity += [w[1]]

    y_pos = np.arange(len(labels), 0, -1)
    axs = fig.add_subplot(211)
    axs.barh(y_pos, popularity, align='center', alpha=0.5, color='#1FFF00')
    axs.grid(color='k', ls='-.', lw=0.25)
    plt.yticks(y_pos, labels)
    plt.xlabel('Usage')
    axs.set_title(f'Favorite Languages in {title}')

    labels = []
    popularity = []
    for i, w in enumerate(y):
        if i > 10:
            break
        labels += [w[0]]
        popularity += [w[1]]

    axs = fig.add_subplot(212)
    axs.set_title(f'Favorite knowledge in {title}')
    axs.bar(labels, popularity)
    plt.setp(axs.get_xticklabels(), rotation=30, horizontalalignment='right')


class City:
    def __init__(self, df: pd.DataFrame, name):
        self.df = df
        self.name = name
        self.number_of_advertise = self.df.shape[0]

    def trend(self):
        entries = self.df['key words'].values.tolist()
        # print(entries)
        return find_common(entries)


def city_analyze(df: pd.DataFrame):
    name_of_cities = [city for i, city in enumerate(df['city'].unique()) if city is not np.nan]

    cities = []

    for c in name_of_cities:
        cities.append(City(df[df['city'] == c], c))

    programming_language = {}

    for j, city in enumerate(cities):
        if j < 25:
            continue
        print(city.name)
        f_lang, f_frame = city.trend()
        tmp = []
        for f in f_frame:
            if f[0] not in languages:
                tmp += [f]

        favorite_languages = []
        for f in f_lang:
            if f[0] in languages:
                favorite_languages += [f]

        draw(favorite_languages, tmp, city.name)
        plt.show()


def country_analyze(df: pd.DataFrame):
    iran = City(df, 'Iran')
    f_lang, f_frame = iran.trend()
    tmp = []
    for f in f_frame:
        if f[0] not in languages:
            tmp += [f]

    favorite_languages = []
    for f in f_lang:
        if f[0] in languages:
            favorite_languages += [f]

    draw(favorite_languages, tmp, iran.name)
    plt.show()


df = pd.read_excel('datasets/dataset.xlsx', engine='openpyxl')
df = df.loc[:, :'kyw']
df.rename(columns={'g': 'name', 'a Type': 'type', 'a Time': 'date', 'j Tiltle': 'job title', 'rmtWrk': 'remote',
                   'kyw': 'key words'}, inplace=True)

country_analyze(df)
city_analyze(df)

"""ads = [city.number_of_advertise for city in cities]
fig, ax = plt.subplots()
ax.grid(True)
ax.bar(name_of_cities, ads, color='g')
xlocs, xlabs = plt.xticks()
for i, v in enumerate(ads):
    plt.text(xlocs[i] - 0.25, v + 10, str(v))

plt.xlabel(f'Cities: {len(cities)}')
plt.ylabel('Number of jobs')
fig.autofmt_xdate()
plt.title('This chart illustrate the number of job request per city')
plt.show()"""

"""import numpy as np

j_title = df['job title'].values.tolist()
x = []
y = []
for title in df['job title'].unique():
    if title is not np.nan:
        # print(f'{type(title)}: {title}: {j_title.count(title)}')
        x += [title]
        y += [j_title.count(title)]
plt.bar(x, y, width=30)
plt.title('Job Title')
plt.show()"""
