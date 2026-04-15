from src.data.dataset_loader import DatasetLoader
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureBuilder
from src.models.train_xgb_cv import cross_validate_xgb
from src.models.train_final import train_final_model, save_artifacts


def main():
    loader = DatasetLoader("data/tig_meta_data.csv")
    df = loader.load()
    loader.validate()

    preprocessor = DataPreprocessor()
    preprocessor.fit(df)
    df_clean = preprocessor.process(df)

    builder = FeatureBuilder()
    X, y = builder.fit_transform(df_clean)

    cross_validate_xgb(X, y)

    model = train_final_model(X, y)

    save_artifacts(model, builder)


if __name__ == "__main__":
    main()