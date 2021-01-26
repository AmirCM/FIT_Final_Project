import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

frame_lan = {'python': ['flask', 'django', 'cherrypy', 'falcon'],
             'javascript': ['angular', 'jquery', 'vue', 'react'],
             'html': ['bootstrap', 'html5', 'foundation', 'skeleton'],
             'css': ['bootstrap', 'html5', 'css3', 'foundation', 'skeleton'],
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
        if i > 20:
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
        labels += [w[0]]
        popularity += [w[1]]

    axs = fig.add_subplot(212)
    axs.set_title(f'Favorite knowledge in {title}')
    axs.bar(labels, popularity)
    plt.setp(axs.get_xticklabels(), rotation=30, horizontalalignment='right')


class Data:
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
        cities.append(Data(df[df['city'] == c], c))

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
    iran = Data(df, 'Iran')
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


def global_job_situation(df: pd.DataFrame):
    j_title = [j for j in df['job title'].values.tolist() if j
               is not np.nan]

    x = []
    y = []
    j_title = nltk.FreqDist(j_title)

    draw(j_title.most_common(), j_title.most_common(15), 'Job Titles')
    plt.show()


def type_analyze(df: pd.DataFrame):
    pr_df = df[df['type'] == 'pr']
    gov_df = df[df['type'] == 'gov']
    print(f'Number of Private company: {pr_df.shape[0]}')
    print(f'Number of Government company: {gov_df.shape[0]}')
    slices = [gov_df.shape[0], pr_df.shape[0]]
    activities = ['Government', 'Private']
    cols = ['#fdf344', '#8126ca']

    plt.pie(slices,
            labels=activities,
            colors=cols,
            startangle=90,
            autopct='%1.1f%%')

    plt.title('Type Graph')

    lan, knw = Data(pr_df, 'Private Company').trend()
    knw = [k for k in knw[:50] if k[0] not in languages]
    lan = [l for l in lan[:50] if l[0] in languages]

    draw(lan, knw[:20], 'Private Company')
    plt.show()


data_f = pd.read_excel('datasets/dataset.xlsx', engine='openpyxl')
data_f = data_f.loc[:, :'kyw']
data_f.rename(columns={'g': 'name', 'a type': 'type', 'a time': 'date', 'j title': 'job title', 'rmtWrk': 'remote',
                   'kyw': 'key words'}, inplace=True)

# country_analyze(data_f)
# city_analyze(data_f)
# global_job_situation(data_f)
type_analyze(data_f)

"""for title in data_f['job title'].unique():
    if title is not np.nan:
        print(f'{type(title)}: {title}: {j_title.count(title)}')
        x += [title]
        y += [j_title.count(title)]
plt.bar(x, y, width=30)
plt.title('Job Title')
plt.show()"""
