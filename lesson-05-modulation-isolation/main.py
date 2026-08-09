import logging
from pathlib import Path

from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.data.save import save_data
from src.data.train_and_test_split import split_into_train_and_test
from src.data.validation import validate_data
from src.features.engineering import add_wind_humidity_ratio
from src.models.evaluate import evaluate_model
from src.models.serialize import serialize_model
from src.models.training import train_model
from src.utils.config import load_config
from src.utils.logger import setup_logger

if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Запуск обучения модели...")

    logger.info("Загрузка конфигурации...")
    config = load_config()
    logger.info("Конфигурация загружена.")

    logger.info("Загрузка данных...")
    raw_data = load_data(config["data"]["raw"])
    logger.info("Данные загружены.")
    required_columns = list(
        set(config["features"]["selected"]) - set(config["features"]["engineered"])
        | {config["target"]}
    )

    logger.info("Валидация данных...")
    validate_data(raw_data, required_columns=required_columns)
    logger.info("Валидация данных завершена.")

    logger.info("Очистка и обработка данных...")
    cleaned_data = clean_data(raw_data)
    engineered_data = add_wind_humidity_ratio(
        cleaned_data, **config["features"]["new_features"]["wind_humidity_ratio"]
    )
    logger.info("Сохранение обработанных данных...")
    save_data(engineered_data, path=Path(config["data"]["processed"]))
    logger.info("Обработанные данные сохранены.")

    logger.info("Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = split_into_train_and_test(
        engineered_data,
        target_column=config["target"],
        feature_columns=config["features"]["selected"],
        random_state=config["model"]["random_state"],
        test_size=config["model"]["test_size"],
    )

    logger.info("Обучение модели...")
    model = train_model(X_train, y_train)
    logger.info("Модель обучена.")

    logger.info("Сериализация модели...")
    serialize_model(model, file_path=Path(config["model"]["output_path"]))
    logger.info("Модель сериализована.")

    logger.info("Оценка модели...")
    evaluate_model(model, X_test, y_test)
