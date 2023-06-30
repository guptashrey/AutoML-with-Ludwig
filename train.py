import pandas as pd
import click
import logging
from ludwig.api import LudwigModel
from ludwig.visualize import confusion_matrix
from ludwig.visualize import learning_curves
from sklearn.model_selection import train_test_split


@click.command
@click.option(
    "--data-file-path",
    "-f",
    type=click.Path(exists=True),
    help="Path to data file to train the classifier",
    required=True,
)
@click.option(
    "--config-file-path",
    "-c",
    type=click.Path(exists=True),
    help="Path to Ludwig config file",
    required=True,
)
def main(data_file_path, config_file_path):
    if data_file_path is None:
        print("[INFO] Please provide the data file")
        return

    if config_file_path is None:
        print("[INFO] Please provide the Ludwig config file")
        return

    data_df = pd.read_json(data_file_path, lines=True)
    train_df, test_df = train_test_split(
        data_df, test_size=0.2, random_state=0, stratify=data_df[["label"]]
    )

    model = LudwigModel(config=config_file_path, logging_level=logging.INFO)

    train_stats, preprocessed_data, output_directory = model.train(dataset=train_df)
    test_stats, predictions, output_directory = model.evaluate(
        test_df, collect_predictions=True, collect_overall_stats=True
    )

    confusion_matrix(
        [test_stats],
        model.training_set_metadata,
        "label",
        top_n_classes=[6],
        model_names=[""],
        normalize=True,
        output_directory="./",
    )
    learning_curves(train_stats, output_feature_name="label", output_directory="./")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
