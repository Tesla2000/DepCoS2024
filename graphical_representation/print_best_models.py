import re
from dataclasses import dataclass
from itertools import groupby, permutations
from pathlib import Path
from statistics import mean

from more_itertools import map_reduce
from numpy import std
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, f_oneway, ttest_rel, wilcoxon

from graphical_representation.result import Result


if __name__ == '__main__':
    top_n_results = 5
    results = tuple(map(Result.from_file_name, Path(
        'file_names').read_text().splitlines()))
    model_divided_results = map_reduce(results, lambda result: result.model)
    for model_name, results in model_divided_results.items():
        table_str = r"""\multirow{5}{*}{ResNet-18} & multichannel & slicing & augmentation & vowel & f1 \\ \cline{2-6}
 & multichannel & slicing & augmentation & vowel & f1 \\ \cline{2-6}
 & multichannel & slicing & augmentation & vowel & f1 \\ \cline{2-6}
 & multichannel & slicing & augmentation & vowel & f1 \\ \cline{2-6}
 & multichannel & slicing & augmentation & vowel & f1 \\ \hline""".replace("ResNet-18", model_name)
        for result in sorted(results, key=lambda item: item.f1, reverse=True)[:top_n_results]:
            table_str = table_str.replace("multichannel", "Yes" if result.multichannel else "No", 1)
            table_str = table_str.replace("slicing", "Yes" if result.window else "No", 1)
            table_str = table_str.replace("augmentation", " ".join(map(str.capitalize, result.augmentation.split())), 1)
            table_str = table_str.replace("vowel", result.vowel, 1)
            table_str = table_str.replace("f1", str(result.f1), 1)
        print(table_str)
