import argparse

import numpy as np
import tensorflow.compat.v1 as tf

from utils.evaluation import Evaluator, open_npz_array, _compute_metrics


def polynomial_mmd_averages(codes_g, codes_r, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / codes_g.shape[1]
    k_xx = (gamma * (codes_g @ codes_g.T) + coef0) ** degree
    k_yy = (gamma * (codes_r @ codes_r.T) + coef0) ** degree
    k_xy = (gamma * (codes_g @ codes_r.T) + coef0) ** degree
    n = codes_g.shape[0]
    m = codes_r.shape[0]
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_npz", required=True)
    parser.add_argument("--ref_npz", required=True)
    args = parser.parse_args()

    fid, sfid, is_score, prec, recall = _compute_metrics(args.ref_npz, args.sample_npz)

    config = tf.ConfigProto(allow_soft_placement=True)
    evaluator = Evaluator(tf.Session(config=config))
    evaluator.warmup()

    with open_npz_array(args.sample_npz, "arr_0") as sample_reader:
        sample_feats, _ = evaluator.compute_activations(sample_reader.read_batches(evaluator.batch_size))
    with open_npz_array(args.ref_npz, "arr_0") as ref_reader:
        ref_feats, _ = evaluator.compute_activations(ref_reader.read_batches(evaluator.batch_size))

    kid = polynomial_mmd_averages(sample_feats, ref_feats)

    print({
        "FID": fid,
        "sFID": sfid,
        "IS": is_score,
        "Precision": prec,
        "Recall": recall,
        "KID": kid,
    })


if __name__ == "__main__":
    main()
