import os
from pathlib import Path

from matplotlib import pyplot as plt
from more_itertools import map_reduce

from Config import Config
from graphical_representation.result import Result

if __name__ == '__main__':
    results = tuple(map(Result.from_file_name, Path(
        'file_names').read_text().splitlines()))
    model_divided_results = map_reduce(results, lambda result: result.model, lambda result: result.f1)
    plt.boxplot(model_divided_results.values(), patch_artist=True)
    plt.xticks(range(1, 1 + len(model_divided_results.values())), list(model_divided_results.keys()))
    plt.ylabel('F1-score')
    result_path = Config.image_path / "model_results.png"
    plt.savefig(result_path)
    os.system(f"convert {result_path.absolute()} -trim {result_path.absolute()}")
