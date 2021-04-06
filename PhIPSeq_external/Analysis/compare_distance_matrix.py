import os
import pandas
import numpy
import time

from scipy.stats import spearmanr, percentileofscore
import scipy.spatial.distance

import matplotlib.pyplot as plt

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

TYPE = ['exist', 'fold'][0]
lib = "agilent"
MIN_OLIGOS = [5, 10, 20]
NUM_PERMS = 1000

if __name__ == '__main__':
    res_path = os.path.join(base_path, "MB_distance")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    chache_path = os.path.join(base_path, "Cache")
    if TYPE == 'exist':
        df = pandas.read_pickle(os.path.join(chache_path, "exist_agilent.pkl"))
        df.fillna(0, inplace=True)
    elif TYPE == 'fold':
        df = pandas.read_pickle(os.path.join(chache_path, "fold_agilent.pkl"))
        df.fillna(1, inplace=True)
        df[df < 1] = 1
        df = numpy.log10(df)
    df.index = df.index.get_level_values(0)
    df.columns = df.columns.get_level_values(0)

    MB_df = pandas.read_pickle(os.path.join(chache_path, "MB_agilent_above200.pkl"))
    inds = MB_df.index.intersection(df.columns)
    MB_df = MB_df.reindex(inds)
    df = df[inds]

    MB_df[MB_df == numpy.min(MB_df)] = 0
    MB_df.fillna(0, inplace=True)

    MB_BrayCurtis = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(MB_df, metric='braycurtis'))

    res = []
    for metric in ['correlation', 'hamming', 'jaccard']:
        for m in MIN_OLIGOS:
            print(m, metric)
            df_m = df[(df != 0).sum(1) >= m]
            print(df_m.shape)
            oligo_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(df_m.T, metric=metric))
            print(oligo_dist.shape)

            sc = spearmanr(MB_BrayCurtis.flatten(), oligo_dist.flatten())
            res.append([sc[0]])

            plt.scatter(MB_BrayCurtis.flatten(), oligo_dist.flatten(), color='#fecd30')
            plt.xlabel("Bray Curtis microbiome distance")
            plt.ylabel("%s oligo distance" % metric)
            plt.title("Base spearman correlation of %g" % sc[0])
            plt.savefig(os.path.join(res_path, "scatter_%s_%d_%s.png") % (TYPE, m, metric))
            plt.close('all')

            for i in range(NUM_PERMS):
                if (i % 100) == 0:
                    print("At %d of %d" % (i, NUM_PERMS), time.ctime())
                perm = numpy.random.permutation(len(oligo_dist))
                res[-1].append(spearmanr(MB_BrayCurtis.flatten(), oligo_dist[perm, :][:, perm].flatten())[0])
            pandas.DataFrame(res).to_csv(os.path.join(res_path, "rand_%d_%s_dist.csv" % (m, TYPE)))

            plt.hist(res[-1][1:], bins=int(NUM_PERMS ** 0.5))
            plt.vlines(res[-1][0], 0, plt.ylim()[1], color='r')
            prc = percentileofscore(res[-1][1:], res[-1][0])
            if prc > 0.5:
                prc = 2 * (100 - prc)
            else:
                prc = 2 * prc
            plt.title("Two sided p_val %g" % (prc / 100))
            plt.xlabel("spearman corr of MB BC with oligo (in >=%d) %s %s" % (m, TYPE, metric))
            plt.savefig(os.path.join(res_path, "hist_%s_%d_%s.png") % (TYPE, m, metric))
            plt.close('all')
