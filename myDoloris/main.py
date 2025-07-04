import os
import pandas as pd
import yaml
from src.data.dataloader import DataLoader
from src.data.label import define_label_binary, define_label_multiclass, LabelEncoder
from src.model.train import train_model_with_val
from src.model.evaluate import evaluate_model
from src.plot import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_avg_scores
)
import time


def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(config):
    df = pd.read_csv(config["data_path"])
    df = LabelEncoder(df)

    if config["label_type"] == "binary":
        df = define_label_binary(df)
        label_col = "label_binary"
    elif config["label_type"] == "multiclass":
        df = define_label_multiclass(df)
        label_col = "label_multiclass"

    loader = DataLoader(
            df=df,
            feature_cols=config["feature_cols"],
            label_col=label_col,
            val_size=config["val_size"],
            test_size=config["test_size"],
            random_state=config["random_state"],
            scale=config["scale"]
        )
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()
    model_name = config["model_name"]
    params = config["all_model_params"][model_name]

    print(f"\nTraining model: {model_name}")
    
    start_time = time.time()
    model, val_metrics = train_model_with_val(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=params
    )
    end_time = time.time() 
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    print("\nValidation Results:", val_metrics)

    test_metrics = evaluate_model(model, X_test, y_test)
    print("\nTest Results:", test_metrics)
    
    conf_matrix = test_metrics["confusion_matrix"]
    report = test_metrics["report"]

    plot_confusion_matrix(conf_matrix, class_names=["Not At Risk", "At Risk"])
    plot_classification_report(report, title="Test Set Classification Report")
    plot_avg_scores(report)


if __name__ == "__main__":
    config = load_config("config.yaml")
    main(config)