import argparse
import os
import matplotlib
import glob
import tqdm
import matplotlib.pyplot as plt 
import numpy as np

from os.path import exists as E
from os.path import join as J

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return [sorted(a[i:i+n])[n//2] for i in range(0, a.shape[0] - n)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory",type=str, help="The experiment output directory")
    args = parser.parse_args()

    assert E(args.output_directory), "Output directory {} does not exist".format(args.output_directory)

    experiments = glob.glob(J(args.output_directory, "*"))

    for ex in tqdm.tqdm(experiments):
        losses = np.load(J(ex, "losses.npy"))
        test_perplexity, test_acc = zip(*np.load(J(ex, "test_performance.npy")))
        train_perplexity, train_acc = zip(*np.load(J(ex, "train_performance.npy")))

        plt.plot (moving_average(losses, 100))
        
        plt.title("Loss")
        plt.savefig(J(ex, 'losses.png'))
        plt.clf()
        plt.figure()
        plt.plot(train_perplexity, label="Train")
        plt.plot(test_perplexity, label="Test")
        plt.legend()
        plt.title("Perplexity")
        plt.savefig(J(ex, 'perplexity.png'))
        plt.clf()

        plt.figure()
        plt.plot(train_acc, label="Train")
        plt.plot(test_acc, label="Test")
        plt.legend()
        plt.title("Accuracy")
        plt.savefig(J(ex, 'accuracy.png'))
        plt.clf()
        

                # np.save(J(experiment_directory, 'losses.npy'), losses)
                #     np.save(J(experiment_directory, 'test_perplexity.npy'), test_perplexity)
                #     np.save(J(experiment_directory, 'train_perplexity.npy'), train_perplexity)
                


if __name__ == '__main__':
    main()