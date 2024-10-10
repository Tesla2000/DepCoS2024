from scipy.stats import mannwhitneyu, shapiro, ttest_ind, ttest_rel, wilcoxon


def calc_p_value(values1, values2):
    abnormal = (shapiro(values1).pvalue < .05 or shapiro(values2).pvalue < .05)
    if len(values1) == len(values2):
        return (wilcoxon if abnormal else ttest_rel)(values1, values2,
                                                        alternative="greater").pvalue
    else:
        return (mannwhitneyu if abnormal else ttest_ind)(values1, values2,
                                                            alternative="greater").pvalue