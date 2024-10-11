import os
from pathlib import Path

from matplotlib import pyplot as plt
from more_itertools import map_reduce

from Config import Config
from graphical_representation.result import Result

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    # Load results and organize by model
    results = tuple(
        map(Result.from_file_name, Path('file_names').read_text().splitlines()))
    model_divided_results = map_reduce(results, lambda result: result.model,
                                       lambda result: result.f1)

    # Prepare data for plotting
    model_names = list(model_divided_results.keys())
    model_f1_scores = list(model_divided_results.values())

    # Plot boxplot
    plt.boxplot(model_f1_scores, patch_artist=True)

    # Calculate means and standard deviations for each model's F1-scores
    means = [np.mean(f1_scores) for f1_scores in model_f1_scores]
    std_devs = [np.std(f1_scores) for f1_scores in model_f1_scores]

    # Set x-ticks and labels
    plt.xticks(range(1, 1 + len(model_f1_scores)), model_names, rotation=-15)
    plt.ylabel('F1-score')

    # Annotate each point with mean and standard deviation values
    for i, (mean, std_dev) in enumerate(zip(means, std_devs), start=1):
        plt.text(i, mean, f'{mean:.2f} ± {std_dev:.2f}', ha='center',
                 va='bottom', fontsize=8, color='black', fontweight='bold')


    # Save the plot
    result_path = Config.image_path / "model_results.png"
    plt.savefig(result_path)

    # Optional: Trim the image with ImageMagick if necessary
    os.system(
        f"convert {result_path.absolute()} -trim {result_path.absolute()}")
