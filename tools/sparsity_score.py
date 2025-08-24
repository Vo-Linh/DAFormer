from numpy import count_nonzero
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--show-dir', default='./work_dirs/tsne_vis', help='Directory to save results')
args = parser.parse_args()

feats_all = []

if os.path.exists(os.path.join(args.show_dir, f'feats_urban.npy')):
    feats_rural = np.load(os.path.join(args.show_dir, f'feats_rural.npy'))
    feats_urban = np.load(os.path.join(args.show_dir, f'feats_urban.npy'))
    print(feats_rural.shape, feats_urban.shape)
    feats_all.append(feats_urban)
    feats_all.append(feats_rural)

print(feats_rural[0][-1])
threshold = 0.05
def count_ge_threshold(array, threshold):
    return np.count_nonzero(np.abs(array) >= threshold)

rural_sparsity = 1 - count_ge_threshold(feats_rural, threshold) / feats_rural.size
urban_sparsity = 1 - count_ge_threshold(feats_urban, threshold) / feats_urban.size
print(f"Rural sparsity: {rural_sparsity}, Urban sparsity: {urban_sparsity}")
feats_all = np.concatenate(feats_all, axis=0)
sparsity = 1 -count_ge_threshold(feats_all, threshold) / feats_all.size
print(f"Total sparsity: {sparsity}")