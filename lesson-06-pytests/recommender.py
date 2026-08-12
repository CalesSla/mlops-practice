import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class RecommenderModel:
    """
    Рекомендательная модель для музыкальных треков на основе жанров и числовых признаков.
    Использует косинусное сходство для вычисления рекомендаций.

    Attributes:
        data_encoded (pd.DataFrame): Датафрейм с закодированными признаками треков.
        similarity_matrix (np.ndarray): Матрица сходства между треками, вычисленная на основе признаков.
    """
    def __init__(self):
        """
        Инициализирует экземпляр RecommenderModel.
        Инициализирует атрибуты data_encoded и similarity_matrix как None.
        """
        self.data_encoded = None
        self.similarity_matrix = None

    def fit(self, df: pd.DataFrame, df_year: pd.DataFrame):
        """
        Метод для обучения модели на предоставленных данных.
        Объединяет данные о треках с данными о годах, кодирует категориальные признаки, 
        нормализует числовые признаки и вычисляет матрицу сходства между треками.

        Args:
            df (pd.DataFrame): Датафрейм с информацией о треках, включая жанры и числовые признаки.
            df_year (pd.DataFrame): Датафрейм с информацией о годах выпуска треков.

        Returns:
            None: Метод изменяет состояние объекта, устанавливая атрибуты data_encoded и similarity_matrix.
        """
        df_year = df_year[["id", "year"]].copy()
        df_year["track_id"] = df_year["id"]
        df_year.drop(columns="id", inplace=True)

        df = pd.merge(df, df_year, on="track_id")

        xtab_song = pd.crosstab(df["track_id"], df["track_genre"]) * 2
        xtab_song.reset_index(inplace=True)

        df_distinct = (
            df.drop_duplicates("track_id")
            .sort_values("track_id")
            .reset_index(drop=True)
        )
        data_encoded = pd.concat(
            [df_distinct, xtab_song.drop(columns=["track_id"])], axis=1
        )

        numerical_features = [
            "explicit",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "year",
        ]
        scaler = MinMaxScaler()
        data_encoded[numerical_features] = scaler.fit_transform(
            data_encoded[numerical_features]
        )

        self.data_encoded = data_encoded
        self.similarity_matrix = cosine_similarity(
            data_encoded[numerical_features + list(xtab_song.columns[1:])]
        )

    def recommend(self, track_title: str, N: int = 5):
        """
        Метод для получения рекомендаций треков, похожих на указанный трек.
        Использует косинусное сходство для нахождения N наиболее похожих треков.

        Args:
            track_title (str): Название трека, для которого нужно получить рекомендации.
            N (int): Количество рекомендаций, которые нужно вернуть. По умолчанию 5.

        Returns:
            list[dict]: Список словарей с информацией о рекомендованных треках, включая название,
            исполнителей и альбом. Пустой список возвращается, если указанный трек не найден в данных.
        """
        indices = pd.Series(
            self.data_encoded.index, index=self.data_encoded["track_name"]
        ).drop_duplicates()

        if track_title not in indices:
            return []

        idx = indices[track_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : N + 1]

        song_indices = [i[0] for i in sim_scores]
        recommended = self.data_encoded[["track_name", "artists", "album_name"]].iloc[
            song_indices
        ]

        return recommended.to_dict(orient="records")


def train_and_save_model():
    """
    Тренирует модель рекомендаций и сохраняет её в файл "model.pkl".

    Returns:
        None
    """
    df = pd.read_csv("dataset.csv").drop(columns="Unnamed: 0")
    df_year = pd.read_csv("data.csv")
    model = RecommenderModel()
    model.fit(df, df_year)
    joblib.dump(model, "model.pkl")


def load_model():
    """
    Загружает модель рекомендаций из файла "model.pkl".

    Returns:
        RecommenderModel: Загруженная модель рекомендаций.
    """
    return joblib.load("model.pkl")


def recommend_songs(model, track_title: str, N: int = 5):
    """
    Получает рекомендации треков на основе обученной модели.

    Args:
        model (RecommenderModel): Обученная модель рекомендаций.
        track_title (str): Название трека, для которого нужно получить рекомендации.
        N (int): Количество рекомендаций, которые нужно вернуть. По умолчанию 5.
    
    Returns:
        list[dict]: Список словарей с информацией о рекомендованных треках, включая название,
        исполнителей и альбом. Пустой список возвращается, если указанный трек не найден в данных.
    """
    return model.recommend(track_title, N)