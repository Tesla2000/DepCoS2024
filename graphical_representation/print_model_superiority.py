import re
from dataclasses import dataclass
from itertools import groupby, permutations
from pathlib import Path
from statistics import mean

from more_itertools import map_reduce
from numpy import std
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, f_oneway, ttest_rel, wilcoxon

from graphical_representation.calc_p_value import calc_p_value
from graphical_representation.result import Result


if __name__ == '__main__':
    results = tuple(map(Result.from_file_name, Path(
        'file_names').read_text().splitlines()))
    model_divided_results = map_reduce(results, lambda result: result.model, lambda result: result.f1)
    print(r"\hline Superior Model & Inferior Model \\ \hline")

    anova_result = f_oneway(*model_divided_results.values()).pvalue
    if anova_result > 0.05:
        exit()
    sorted_results = sorted(model_divided_results.items(), key=lambda item: -sum(item[1]))
    for index, (model1, value1) in enumerate(sorted_results, 1):
        for model2, value2 in sorted_results[index:]:
            p_value = calc_p_value(value1, value2)
            if p_value > 0.05:
                continue
            print(fr"{model1} & {model2} \\ \hline")
