import os
import argparse

import pandas as pd
import numpy as np


# To parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Evaluate GestaltMatcher-Arc for GMDB v1.1.0')
    parser.add_argument('--set', default='eu',
                        help='Which set is being analyzed? (Options: "eu", "others")')
    return parser.parse_args()


def main():
    # Load parameter arguments
    args = parse_args()

    results_dir = './experiments_anc'

    if args.set == 'eu':
        print("Analyzing for EU+EU*:")
    elif args.set == 'others':
        print("Analyzing for EU+Others:")
    else:
        print(f"Unknown set given (got '--set {args.set}')")
        exit()

    ancs = []
    distrs = []
    top1s = []
    top5s = []
    for i in range(1,6):
        if args.set == 'eu':
            file_path = os.path.join(results_dir, f'performance_s{i}_seed{i}_eu.npy')
        elif args.set == 'others':
            file_path = os.path.join(results_dir, f'performance_s{i+10}_seed{i}_all_anc.npy')

        anc, distr, top1, top5 = np.load(file_path, allow_pickle=True)

        ancs.append(anc)
        distrs.append(distr.astype(float))
        top1s.append(top1.astype(float))
        top5s.append(top5.astype(float))

    # fix order of distribution array
    for i in range(len(ancs)):
        missing_indices = np.where(np.isin(ancs[0], ancs[i], invert=True))[0]
        for mi in missing_indices:
            distrs[i] = np.insert(distrs[i], missing_indices, 0)
            top1s[i] = np.insert(top1s[i], missing_indices, -1)
            top5s[i] = np.insert(top5s[i], missing_indices, -1)

    top1s = np.stack(top1s)
    top5s = np.stack(top5s)
    corr_top1_res = []
    corr_top5_res = []
    for anc, distr, std, top1, top5 in zip(ancs[0], np.mean(distrs, axis=0), np.std(distrs, axis=0), top1s.T, top5s.T):
        corr_top1 = np.sum(top1[top1 > 0]) / len(top1[top1 > 0])
        corr_top5 = np.sum(top5[top5 > 0]) / len(top5[top5 > 0])

        corr_top1_res.append(corr_top1)
        corr_top5_res.append(corr_top5)

        print(f"\t{anc} ({distr} +/- {std:.1f}): "
              f"{corr_top1*100:.2f} +/- {np.std(top1[top1 > 0])*100:.2f}, "
              f"{corr_top5*100:.2f} +/- {np.std(top5[top5 > 0])*100:.2f}")

    print(f"Mean top-1: {np.mean(corr_top1_res)*100:.2f} "
          f"({(np.sum(corr_top1_res) - corr_top1_res[8]) / (len(corr_top1_res)-1)*100:.2f} without mixed ancestries)")
    print(f"Mean top-5: {np.mean(corr_top5_res)*100:.2f} "
          f"({(np.sum(corr_top5_res) - corr_top5_res[8]) / (len(corr_top5_res)-1)*100:.2f} without mixed ancestries)")

if __name__ == '__main__':
    main()
