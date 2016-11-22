import numpy as np
import pandas as pd
import nltk.data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class FactorizeActors(object):

    """
        Clase que se encarga de la factorizacion de los
        nombres de los actores del IMDB Dataset.
    """

    def __init__(self, df, columns):
        """
            Crea un objeto de tipo FactorizeActors
        :param df: Pandas DataFrame
        :param columns: Lista con nombre de columnas de actores
        """
        self.df = df
        self.columns = columns
        self.actors = None

    def unique_actors(self):
        """
            Genera el conjunto de actores que se encuentran
            en las distintas columnas seleccionadas
        """
        tmp = []
        for col in self.columns:
            tmp.append(self.df[col].unique())
        actors = np.concatenate(tmp)
        self.actors = dict(zip(actors, range(0, len(actors))))

    def __call__(self):
        """
            Factoriza las columnas de actores seleccionadas
        :return: Pandas DataFrame con columnas factorizadas
        """
        self.unique_actors()
        for col in self.columns:
            self.df[col] = self.df[col].apply(lambda x: self.actors[x])
        return self.df

class ProcessIMDBData(object):

    """
        Clase que se encarga del pre-procesamineto del
        IMDB Dataset.
    """

    def __init__(self, path_file, drop_cols):
        """
            Crea una instancia de la clase
        :param path_file: Path del archivo CSV con los datos
        :param drop_cols: Arreglo con las nombres de las columnas a eliminar
        """
        self.df = pd.read_csv(path_file)
        self.drop_cols = drop_cols
        self.vect = None

    def group_movies(self, x):
        """
            Determina si una pelicula es Muy buena, Buena o Mala,
            en función de su score.
        :param x: IMDB score
        :return: 0 --> Mala, 1 --> Buena, 2 --> Muy Buena
        """
        if x >=7.5:
            return 2
        elif x<7.5 and x>=6:
            return 1
        else:
            return 0

    def generate_vectorizer(self):
        """
            Genera el vectorizador de palabras claves
        """
        temp = self.df.plot_keywords.str.replace('|',' ')

        nltk.data.path = ['nltk_data/']

        stopw = set(stopwords.words('english'))
        corpus = []
        for e in temp.values:
            if type(e) is str:
                corpus.append(e)
        vocabulary = " ".join(corpus)
        vocabulary = vocabulary.split(" ")
        vocabulary = list(filter(lambda x: not x[0] in stopw, vocabulary))
        vocabulary = set(vocabulary)

        self.vect = TfidfVectorizer(sublinear_tf=True, max_df=2, analyzer='word',
                   stop_words=stopw, vocabulary=vocabulary)

        self.vect.fit(corpus)

    def transform_keywords(self,x):
        """
            Transforma las plot_keywords en un valor float
            que determina la importancia de las mismas dentro
            del vocabulario del DF.
        :param x: Plot Keywords de un pelicula
        :return: Valor de importancia
        """
        if pd.isnull(x):
            return -1
        keywords = x.replace('|'," ")
        doc_tfidf = self.vect.transform([keywords])
        t = doc_tfidf.toarray().sum()
        return t

    def __call__(self):
        """
            Procesa las distintas columnas del DataFrame y elimina las columnas
            que no son necesarias
        """
        self.df['imdb_score'] = self.df['imdb_score'].apply(self.group_movies)
        self.df = FactorizeActors(self.df, ['actor_1_name', 'actor_2_name', 'actor_3_name'])()

        ## Para obtener los indices de los paises y los lenguajes de las peliculas
        self.idx_language, self.label_language = pd.factorize(self.df['language'])
        self.idx_country, self.label_country = pd.factorize(self.df['country'])

        ## Para obtener el indice del director
        self.idx_director, self.label_director = pd.factorize(self.df['director_name'])

        #Para obtener los indices del titulo de las peliculas y el content rating
        self.idx_movie, self.label_movie = pd.factorize(self.df['movie_title'])
        self.idx_content, self.label_content = pd.factorize(self.df['content_rating'])

        self.df = pd.concat([self.df, self.df.genres.str.get_dummies(sep='|')], axis=1)

        ## Para codificar las plot_keywords del
        self.generate_vectorizer()
        self.df['plot_keywords'] = self.df['plot_keywords'].apply(lambda x: self.transform_keywords(x))

        self.df['language'] = self.idx_language
        self.df['country'] = self.idx_country
        self.df['director_name'] = self.idx_director
        self.df['movie_title'] = self.idx_movie
        self.df['content_rating'] = self.idx_content

        self.df.fillna(value=-1, inplace=True)

        for col in self.drop_cols:
            self.df.drop(col, axis=1, inplace=True)

        return self.df
