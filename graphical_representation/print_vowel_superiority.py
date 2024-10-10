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
    model_divided_results = map_reduce(results, lambda result: result.model)
    print(r"\hline Model & Superior Vowel & Inferior Vowel \\ \hline")
    for model_name, results in model_divided_results.items():
        table_str = r"""\multirow{5}{*}{ResNet-18} & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \cline{2-3}
 & multichannel & slicing \\ \hline""".replace("ResNet-18", model_name)
        divided_results = map_reduce(results, lambda result: result.vowel, lambda result: result.f1)
        anova_result = f_oneway(*divided_results.values()).pvalue
        if anova_result > 0.05:
            continue
        sorted_results = sorted(divided_results.items(), key=lambda item: -sum(item[1]))
        for index, (vowel1, value1) in enumerate(sorted_results, 1):
            for vowel2, value2 in sorted_results[index:]:
                p_value = calc_p_value(value1, value2)
                if p_value > 0.05:
                    continue
                table_str = table_str.replace("multichannel", vowel1, 1)
                table_str = table_str.replace("slicing", vowel2, 1)
        tabel_str = table_str.replace("\cline{2-3}\n & multichannel & slicing \\\\ ", "")
        tabel_str = tabel_str.replace("\multirow{5", "\multirow{" + str(tabel_str.count("\cline{2-3}")))
        print(tabel_str)
