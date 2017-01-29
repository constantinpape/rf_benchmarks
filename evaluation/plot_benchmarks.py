import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_train_res(train = True, plot = False, markdn = False):

    res_pth = '../results/benchmarks_training.pkl' if train else '../results/benchmarks_prediction.pkl'
    with open(res_pth) as f:
        train_res = pickle.load(f)

    res_vi2 = train_res['vi2']
    res_vi3 = train_res['vi3']
    res_skl = train_res['sk']

    n_threads = res_vi2.keys()
    n_threads.sort()
    t_vi2 = []
    std_vi2 = []
    t_vi3 = []
    std_vi3 = []
    t_skl = []
    std_skl = []

    for n_thr in n_threads:
        t_vi2.append(res_vi2[n_thr][0])
        std_vi2.append(res_vi2[n_thr][1])

        t_vi3.append(res_vi3[n_thr][0])
        std_vi3.append(res_vi3[n_thr][1])

        t_skl.append(res_skl[n_thr][0])
        std_skl.append(res_skl[n_thr][1])

    if plot:
        plt.figure()
        plt.title("RF-Training-Benchmark" if train else "RF-Prediction-Benchmark")
        plt.errorbar(n_threads, t_vi2, yerr = std_vi2, label = "vigra-rf2")
        plt.errorbar(n_threads, t_vi3, yerr = std_vi3, label = "vigra-rf3")
        plt.errorbar(n_threads, t_skl, yerr = std_skl, label = "sklearn-rf")
        plt.xlabel('number of threads')
        plt.ylabel('training time [s]' if train else 'prediction time [s]')
        plt.legend()
        save_path = './plot_train.png' if train else './plot_prediction.png'
        plt.savefig(save_path)
        plt.close()

    if markdn:
        markdn_path = './markdn_train.md' if train else './markdn_prediction.md'
        with open(markdn_path, 'w') as f:
            f.write("| Num Threads | Vigra RF2 [s] | Vigra RF3 [s] | Sklearn RF [s] | \n")
            f.write("| ----------- | ------------: | ------------: | -------------: | \n")
            for i, n_thr in enumerate(n_threads):
                f.write("| %i        | %.3f +- %.3f | %.3f +- %.3f | %.3f +- %.3f | \n" % (n_thr,
                    t_vi2[i], std_vi2[i], t_vi3[i], std_vi3[i], t_skl[i], std_skl[i]))


if __name__ == '__main__':
    plot_train_res(train = False, plot = True, markdn = True)
