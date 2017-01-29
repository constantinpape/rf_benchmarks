import cPickle as pickle

def mrkdn_table_gridsearch():
    gridsearch_res_path = '../results/benchmarks_gridsearch.pkl'
    with open(gridsearch_res_path) as f:
        gridsearch_res = pickle.load(f)

    keys = gridsearch_res.keys()
    keys.sort()
    keys_resorted = []
    keys_resorted.extend(keys[6:])
    keys_resorted.extend(keys[:6])

    #markdn_path = './markdn_gridsearch_prediction_time.md'
    #with open(markdn_path, 'w') as f:
    #    f.write("| MinLeafSize / | 1   | 2   | 5   | 10  | 15  | 20  | \n")
    #    f.write("| MaxDepth      |     |     |     |     |     |     | \n")
    #    f.write("| ------------- | --: | --: | --: | --: | --: | --: | \n")

    #    for key in keys_resorted:
    #        max_depth = key[0]
    #        min_leaf = key[1]
    #        if min_leaf == 0:
    #            line_str = "| %s             |" % ("None" if max_depth == -1 else str(max_depth))
    #        res = gridsearch_res[key]
    #        line_str += " %.3f +- %.3f s |" % (res[2], res[3])
    #        if min_leaf == 20:
    #            line_str += "\n"
    #            f.write(line_str)

    markdn_path = './markdn_gridsearch_prediction_accuracy.md'
    with open(markdn_path, 'w') as f:
        f.write("| MinLeafSize / | 1   | 2   | 5   | 10  | 15  | 20  | \n")
        f.write("| MaxDepth      |     |     |     |     |     |     | \n")
        f.write("| ------------- | --: | --: | --: | --: | --: | --: | \n")

        for key in keys_resorted:
            max_depth = key[0]
            min_leaf = key[1]
            if min_leaf == 0:
                line_str = "| %s             |" % ("None" if max_depth == -1 else str(max_depth))
            res = gridsearch_res[key]
            line_str += " %.3f +- %.3f |" % (res[4], res[5])
            if min_leaf == 20:
                line_str += "\n"
                f.write(line_str)

if __name__ == '__main__':
    mrkdn_table_gridsearch()
