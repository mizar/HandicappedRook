import numpy as np
from scipy.stats import beta,norm
import sys
args = sys.argv
# max_delta_rating = 2**13 # レーティング差の最大値を表す定数

def infer_rating(win, lose, draw=0): # レーティングの区間推定を行う
    def erorate(winrate): # 勝率からレーティング差を計算する
        if winrate == 1:
            return +np.inf
        if winrate == 0:
            return -np.inf
        return 400.0 * np.log10(winrate / (1 - winrate))

    def clopper_pearson(k, n, alpha): # clopper_pearson法による二項分布近似
        alpha2 = (1 - alpha) / 2
        lower = beta.ppf(alpha2, k, n - k + 1)
        upper = beta.ppf(1 - alpha2, k + 1, n - k)
        return (lower, upper, k, n, alpha)

    def binom_test(k, n, rate):
        sumk = 1
        f, r, s, kk, l1 = (rate / (1 - rate), k, n - k, 1, 1)
        while kk == kk and kk != 0:
            r += 1
            kk = kk * f * s / r
            s -= 1
            sumk += kk
            l1 += kk
        f, r, s, kk, l2 = ((1 - rate) / rate, n - k, k, 1, 0)
        while kk == kk and kk != 0:
            r += 1
            kk = kk * f * s / r
            s -= 1
            sumk += kk
            l2 += kk
        return (l1 / sumk, l2 / sumk)

    def print_result(result): # 結果の出力
        r0, r1, k, n, alpha = result
        sigma = norm.ppf(alpha / 2 + 0.5)
        lowerrate = 0 if r0 != r0 else 100 * r0
        lower = -np.inf if r0 != r0 else erorate(r0)
        # lowerscore = np.nan if r0 != r0 else binom_test(k, n, r0)[0]
        upperrate = 100 if r1 != r1 else 100 * r1
        upper = +np.inf if r1 != r1 else erorate(r1)
        # upperscore = np.nan if r1 != r1 else binom_test(n - k, n, 1 - r1)[0]
        print(f"{sigma:.3f}σ: R({100 * alpha:.7f}%){lower:+8.2f} ({lowerrate:6.2f}%) ～{upper:+8.2f} ({upperrate:6.2f}%), Range:{upper - lower:7.2f}")

    match = win + lose
    print(f"有効試合数: {match}")
    winrate = np.nan if match == 0 else 100 * win / match
    deltarate = np.nan if match == 0 else erorate(win / match)
    drawrate = np.nan if (draw + match) == 0 else 100 * draw / (draw + match)
    print(f"勝率: {winrate:.2f}%, ⊿R: {deltarate:+.2f}, 引分率: {drawrate:.2f}%")
    print_result(clopper_pearson(win, match, 0.25))
    print_result(clopper_pearson(win, match, 0.38292492254802624)) # 0.5σ
    print_result(clopper_pearson(win, match, 0.5))
    print_result(clopper_pearson(win, match, 0.68268949213708585)) # 1.0σ
    print_result(clopper_pearson(win, match, 0.75))
    print_result(clopper_pearson(win, match, 0.80))
    print_result(clopper_pearson(win, match, 0.86638559746228383)) # 1.5σ
    print_result(clopper_pearson(win, match, 0.90))
    print_result(clopper_pearson(win, match, 0.95))
    print_result(clopper_pearson(win, match, 0.95449973610364158)) # 2.0σ
    print_result(clopper_pearson(win, match, 0.98))
    print_result(clopper_pearson(win, match, 0.98758066934844768)) # 2.5σ
    print_result(clopper_pearson(win, match, 0.99))
    print_result(clopper_pearson(win, match, 0.99730020393673979)) # 3.0σ
    print_result(clopper_pearson(win, match, 0.998))
    print_result(clopper_pearson(win, match, 0.999))
    print_result(clopper_pearson(win, match, 0.99953474184192892)) # 3.5σ
    print_result(clopper_pearson(win, match, 0.9998))
    print_result(clopper_pearson(win, match, 0.9999))
    print_result(clopper_pearson(win, match, 0.99993665751633376)) # 4.0σ
    print_result(clopper_pearson(win, match, 0.99998))
    print_result(clopper_pearson(win, match, 0.99999))
    print_result(clopper_pearson(win, match, 0.99999320465375074)) # 4.5σ
    print_result(clopper_pearson(win, match, 0.999998))
    print_result(clopper_pearson(win, match, 0.999999))
    print_result(clopper_pearson(win, match, 0.99999942669685615)) # 5.0σ
    print_result(clopper_pearson(win, match, 0.9999998))
    print_result(clopper_pearson(win, match, 0.9999999))
    print_result(clopper_pearson(win, match, 0.99999996202087527)) # 5.5σ
    print_result(clopper_pearson(win, match, 0.99999998))
    print_result(clopper_pearson(win, match, 0.99999999))
    print_result(clopper_pearson(win, match, 0.999999998))
    print_result(clopper_pearson(win, match, 0.99999999802682460)) # 6.0σ
    print_result(clopper_pearson(win, match, 0.999999999))

infer_rating(int(args[1]), int(args[2]), int(args[3] if len(args) >= 4 else 0))
