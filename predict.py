from ludwig.api import LudwigModel
import click


@click.command
@click.option(
    "--data-file-path",
    "-f",
    type=click.Path(exists=True),
    help="Path to data file to run the inference",
    required=True,
)
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True),
    help="Path to the trained model directory",
    required=True,
)
def main(data_file_path, model_dir):
    model = LudwigModel.load(model_dir)
    predictions, _ = model.predict(dataset=data_file_path)
    predictions = predictions[
        [
            "label_predictions",
            "label_probabilities_0",
            "label_probabilities_1",
            "label_probabilities_2",
            "label_probabilities_3",
            "label_probabilities_4",
            "label_probabilities_5",
        ]
    ]
    predictions.to_csv("predictions.csv")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
