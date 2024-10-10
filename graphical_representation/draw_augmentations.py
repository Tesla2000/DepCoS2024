import os
from pathlib import Path

from matplotlib import pyplot as plt
from more_itertools import map_reduce

from Config import Config
from graphical_representation.result import Result

if __name__ == '__main__':
    results = tuple(map(Result.from_file_name, Path(
        'file_names').read_text().splitlines()))
    augmentation_divided_results = map_reduce(filter(lambda result: not result.window, results), lambda result: result.augmentation, lambda result: result.f1)
    plt.boxplot(augmentation_divided_results.values(), patch_artist=True)
    plt.xticks(range(1, 1 + len(augmentation_divided_results.values())), list(map(str.capitalize, augmentation_divided_results.keys())), rotation=-8)
    plt.ylabel('F1-score')
    result_path = Config.image_path / "augmentation_results.png"
    plt.savefig(result_path)
    os.system(f"convert {result_path.absolute()} -trim {result_path.absolute()}")
