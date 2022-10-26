
from functions import *


def main():

    source_paths, target_paths, plot_path = get_paths()
    device = start_gpu()
    TrainingData = load_data(target_paths[0])
    ValidationData = load_data(target_paths[1])
    train(TrainingData, ValidationData, device, plot_path)


if __name__ == "main":
    main()