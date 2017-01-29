import cPickle as pickle


def mrkdn_table_sklearn_ram():
    sk_res_path = '../results/sklearn_ram.pkl'
    with open(sk_res_path) as f:
        sk_res = pickle.load(f)

    keys = sk_res.keys()
    keys.sort()

    markdn_path = './markdn_sklear_ram.md'
    with open(markdn_path, 'w') as f:
        f.write("| Num Threads / | 1   | 2   | 4   | 8   | 10  | 20  | \n")
        f.write("| Num Trees     |     |     |     |     |     |     | \n")
        f.write("| ------------- | --: | --: | --: | --: | --: | --: | \n")

        for key in keys:
            ntrees   = key[0]
            nthreads = key[1]
            if nthreads == 1:
                line_str = "| %i             |" % ntrees
            res = sk_res[key]
            line_str += " %.2f GB |" % (res / 1000)
            if nthreads == 20:
                line_str += "\n"
                f.write(line_str)


if __name__ == '__main__':
    mrkdn_table_sklearn_ram()
