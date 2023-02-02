import numpy as np

idxpath = '/mnt/ceph/users/cmodi/contrastive/analysis/test-train-splits/'

def split_index(n, test_frac, seed):
    np.random.seed(seed)
    idx = np.random.permutation(np.arange(n))
    split = int((1-test_frac)*n)
    train = idx[:split]
    test = idx[split:]
    fname = f"N{n}-f{test_frac:0.2f}-S{seed}"
    print(train)
    print(test)
    np.save(f"{idxpath}train-{fname}", train)
    np.save(f"{idxpath}test-{fname}", test)


if __name__=="__main__":

    n = 2000
    test_frac = 0.15
    seed = 0
    split_index(n, test_frac, seed)

