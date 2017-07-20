import numpy as np
from scipy.stats import beta
import sys
args = sys.argv
max_delta_rating = 2**15 # レーティング差の最大値を表す定数

def infer_rating(win, lose, draw): # レーティングの区間推定を行う
    def erorate(winrate): # 勝率からレーティング差を計算する
        if winrate == 1:
            return max_delta_rating
        if winrate == 0:
            return -max_delta_rating
        return 400.0 * np.log10(winrate / (1 - winrate))

    def clopper_pearson(k, n, alpha): # clopper_pearson法による二項分布近似
        alpha2 = (1 - alpha) / 2
        lower = beta.ppf(alpha2, k, n - k + 1)
        upper = beta.ppf(1 - alpha2, k + 1, n - k)
        return (lower, upper)
 
    def print_result(result, alpha_string): # 結果の出力
        lowerrate = 0 if result[0] != result[0] else 100 * result[0]
        lower = -np.inf if result[0] != result[0] else erorate(result[0])
        upperrate = 100 if result[1] != result[1] else 100 * result[1]
        upper = np.inf if result[1] != result[1] else erorate(result[1])
        print(f"R({alpha_string}): {lower:+8.2f}({lowerrate:6.2f}%)～{upper:+8.2f}({upperrate:6.2f}%), Range: {upper - lower:8.2f}")

    match = win + lose
    print(f"有効試合数: {match}")
    winrate = np.nan if match == 0 else 100 * win / match
    deltarate = np.nan if match == 0 else erorate(win / match)
    drawrate = np.nan if (draw + match) == 0 else 100 * draw / (draw + match)
    print(f"勝率: {winrate:.2f}%, ⊿R: {deltarate:+.2f}, 引分率: {drawrate:.2f}%")
    print_result(clopper_pearson(win, match, 0.5), "50.0000%")
    print_result(clopper_pearson(win, match, 0.75), "75.0000%")
    print_result(clopper_pearson(win, match, 0.80), "80.0000%")
    print_result(clopper_pearson(win, match, 0.90), "90.0000%")
    print_result(clopper_pearson(win, match, 0.95), "95.0000%")
    print_result(clopper_pearson(win, match, 0.98), "98.0000%")
    print_result(clopper_pearson(win, match, 0.99), "99.0000%")
    print_result(clopper_pearson(win, match, 0.998), "99.8000%")
    print_result(clopper_pearson(win, match, 0.999), "99.9000%")
    print_result(clopper_pearson(win, match, 0.9998), "99.9800%")
    print_result(clopper_pearson(win, match, 0.9999), "99.9900%")
    print_result(clopper_pearson(win, match, 0.999998), "99.9998%")
    print_result(clopper_pearson(win, match, 0.999999), "99.9999%")

infer_rating(int(args[1]), int(args[2]), int(args[3]))
