import re
from itertools import groupby, permutations
from pathlib import Path
from statistics import mean

from numpy import std
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, f_oneway, ttest_rel, wilcoxon

if __name__ == '__main__':
    # print('\n'.join(sorted((' & '.join(map(str, (("ResNet101" if "root" in line else "ResNet18") if "ResNet" in line else "VGG19", "Yes" if "MultiChannel" in line else "No", "Yes" if "Window" in line else "No", re.findall(r'\.([A-Z\_]+)\.', line)[0].replace('_', ' ').lower(), re.findall(r'_([auil]+)_', line)[0], "0." + re.findall(r'f1\_0\.(\d+)', line)[0]))) + '\\\\' for index, line in enumerate(Path('info.log').read_text().splitlines())), key=lambda line: (line.split()[0], -int(line[-4:-2])))))
    all_data = sorted(((
        (("ResNet101" if "root" in line else "ResNet18") if "ResNet" in line else "VGG19"),
        "MultiChannel" in line,
        "Window" in line,
        re.findall(r'\.([A-Z\_]+)\.', line)[0].replace('_', ' ').lower(),
        re.findall(r'_([auil]+)_', line)[0], int(re.findall(r'f1\_0\.(\d+)', line)[0])
    ) for line in Path('info_no_ResNet101.log').read_text().splitlines()), key=lambda item: item[0])
    model_divided_data = dict((key, tuple(value)) for key, value in groupby(all_data, key=lambda item: item[0]))
    print("Models:", f_oneway(*tuple(tuple(item[-1] for item in model) for model in model_divided_data.values())).pvalue)
    for (model1, values1), (model2, values2) in permutations(model_divided_data.items(), 2):
        values1 = tuple(map(lambda item: item[-1], values1))
        values2 = tuple(map(lambda item: item[-1], values2))
        anormal = (shapiro(values1).pvalue < .05 or shapiro(values2).pvalue < .05)
        p_value = (wilcoxon if anormal else ttest_rel)(values1, values2, alternative="greater").pvalue
        print(f"{model1} $\mean$={mean(values1):.2f}, $\sigma$={std(values1):.2f}")
        print(f"{model2} $\mean$={mean(values2):.2f}, $\sigma$={std(values2):.2f}")
        if p_value < .05:
            # print("Anormality", anormal)
            print(f"{model1} better than {model2} {p_value:.2e}")
    model_divided_data["All"] = all_data

    for model in model_divided_data:
        print()
        print(model)
        model_data = model_divided_data[model]
        x, y = tuple(item[-1] for item in model_data if item[1]), tuple(item[-1] for item in model_data if not item[1])
        print("Multichannel values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
        x, y = tuple(item[-1] for item in model_data if item[2]), tuple(item[-1] for item in model_data if not item[2])
        print("Window values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
        x, y = tuple(item[-1] for item in model_data if len(item[4]) == 3), tuple(item[-1] for item in model_data if not len(item[4]) == 3)
        print("Multivowel values higher:", (mannwhitneyu if (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05) else ttest_ind)(x, y, alternative="greater").pvalue)
        augmentations = tuple(tuple(item[-1] for item in value) for key, value in groupby(sorted(model_data, key=lambda item: item[3]), lambda item: item[3]))
        print("Augmentations:", f_oneway(*augmentations).pvalue)
        vowels = dict((key, tuple(item[-1] for item in value)) for key, value in
                      groupby(sorted(model_data, key=lambda item: item[4]), lambda item: item[4]))
        p_value_vowels = f_oneway(*vowels.values()).pvalue
        print("Vowels:", p_value_vowels)
        if p_value_vowels > .05:
            continue
        for (vowel1, values1), (vowel2, values2) in permutations(vowels.items(), 2):
            anormal = (shapiro(x).pvalue < .05 or shapiro(y).pvalue < .05)
            p_value = (wilcoxon if anormal else ttest_rel)(values1, values2, alternative="greater").pvalue
            if p_value < .05:
                # print("Anormality", anormal)
                print(f"\item /{vowel1}/ better than /{vowel2}/ p\_value={p_value:.2e},")
    pass
