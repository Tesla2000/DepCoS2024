import re
from dataclasses import dataclass
from itertools import groupby, permutations
from pathlib import Path
from statistics import mean

from more_itertools import map_reduce
from numpy import std
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, f_oneway, ttest_rel, wilcoxon

@dataclass
class Result:
    f1: float
    model: str
    multichannel: bool
    window: bool
    vowel: str
    augmentation: str

    @classmethod
    def from_file_name(cls, file_name: str) -> "Result":
        return Result(
            f1=float(re.findall(r'f1\_(0\.\d+)', file_name)[0]),
            multichannel='MultiChannel' in file_name,
            window='Window' in file_name,
            vowel=re.findall(r'\_([auil]+)_', file_name)[0],
            augmentation=re.findall(r'\.([A-Z\_]+)\.pth', file_name)[0].replace('_', ' '),
            model=re.findall(r'(Window|Traditional)([A-Za-z]+)\_', file_name)[0][1]
        )

def divide2sets_and_compare(divided_results: dict[str, list[float]]):
    print("ANOVA:", f_oneway(*divided_results.values()).pvalue)
    for (name1, values1), (name2, values2) in permutations(divided_results.items(), 2):
        abnormal = (shapiro(values1).pvalue < .05 or shapiro(values2).pvalue < .05)
        if len(values1) == len(values2):
            p_value = (wilcoxon if abnormal else ttest_rel)(values1, values2, alternative="greater").pvalue
        else:
            p_value = (mannwhitneyu if abnormal else ttest_ind)(values1, values2, alternative="greater").pvalue
        if p_value < .05:
            print("Abnormality", abnormal)
            print(f"{name1} better than {name2} {p_value:.2e}")
    for name, values in divided_results.items():
        print(f"{name} $\mean$={mean(values):.2f}, $\sigma$={std(values):.2f}")

if __name__ == '__main__':
    print(100 * "=" + f"\nModel comparison:\n\n")
    results = tuple(map(Result.from_file_name, Path('file_names').read_text().splitlines()))
    model_divided_results = map_reduce(results, lambda result: result.model, lambda result: result.f1)
    divide2sets_and_compare(model_divided_results)
    model_divided_data = map_reduce(results, lambda result: result.model)
    for model, data in (*model_divided_data.items(), ("All models", results)):
        print(100*"=" + f"\nFor {model}:\n\n")
        for divisor in ("multichannel", "window", "vowel", "augmentation"):
            print(divisor.capitalize(), "comparison:\n")
            divided_data = map_reduce(data, lambda item: getattr(item, divisor), lambda result: result.f1)
            divide2sets_and_compare(divided_data)

    #     x, y = tuple(item[-1] for item in model_data if item[1]), tuple(item[-1] for item in model_data if not item[1])
    #     print("Multichannel values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
    #     x, y = tuple(item[-1] for item in model_data if item[2]), tuple(item[-1] for item in model_data if not item[2])
    #     print("Window values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
    #     x, y = tuple(item[-1] for item in model_data if len(item[4]) == 3), tuple(item[-1] for item in model_data if not len(item[4]) == 3)
    #     print("Multivowel values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
    #     augmentations = tuple(tuple(item[-1] for item in value) for key, value in groupby(sorted(model_data, key=lambda item: item[3]), lambda item: item[3]))
    #     print("Augmentations:", f_oneway(*augmentations).pvalue)
    #     vowels = dict((key, tuple(item[-1] for item in value)) for key, value in
    #                   groupby(sorted(model_data, key=lambda item: item[4]), lambda item: item[4]))
    #     p_value_vowels = f_oneway(*vowels.values()).pvalue
    #     print("Vowels:", p_value_vowels)
    #     if p_value_vowels > .05:
    #         continue
    #     for (vowel1, values1), (vowel2, values2) in permutations(vowels.items(), 2):
    #         anormal = (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05)
    #         p_value = (wilcoxon if anormal else ttest_rel)(values1, values2, alternative="greater").pvalue
    #         if p_value < .05:
    #             # print("Anormality", anormal)
    #             print(f"\item /{vowel1}/ better than /{vowel2}/ p\_value={p_value:.2e},")
    # pass
