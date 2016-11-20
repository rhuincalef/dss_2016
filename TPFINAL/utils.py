import numpy as np

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

    def factorize(self):
        """
            Factoriza las columnas de actores seleccionadas
        :return: Pandas DataFrame con columnas factorizadas
        """
        self.unique_actors()
        for col in self.columns:
            self.df[col] = self.df[col].apply(lambda x: self.actors[x])
        return self.df
