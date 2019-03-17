import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ture_positive_rates = [1, 1, 1]
    false_positive_rates = [1, 0.0479088, 0.00437382]

    for index in range(len(false_positive_rates)):
        plt.scatter(false_positive_rates[index], ture_positive_rates[index], marker='x', label="stage"+str(index))


    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1.5)
    plt.ylim(0, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/plot.png")
    # plt.show()
