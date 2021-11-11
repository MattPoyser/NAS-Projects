import os
import matplotlib.pyplot as plt


def main(file):
    hards = []
    masters = []
    accs = []

    with open(file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            try:
                print(lines[i])
                try:
                    print(lines[i+1])
                except IndexError:
                    break
                print("next")
                name = lines[i]
                acc = lines[i+1]
                name_splits = name.split("aa")
                subset_size = name_splits[0][name_splits[0].index("1"):]
                acc_splits = acc.split("=")
                if len(acc_splits) == 1:
                    continue
                hardness = name_splits[1]
                mastery = name_splits[2][:-5]
                print(acc_splits)
                final_acc = acc_splits[-1]
                hards.append(float(hardness))
                masters.append(float(mastery))
                accs.append(float(final_acc))
            except ValueError:
                print("skipping line", lines[i])

    max_hard = max(hards)
    min_hard = min(hards)
    max_master = max(masters)
    min_master = min(masters)
    new_hards = [hard/max_hard for hard in hards]
    new_masters = [master/max_master for master in masters]
    combine = [((new_hards[i] + new_masters[i]) / 2) for i in range(len(new_masters))]
    plt.plot(hards, accs, 'r+')
    plt.plot(masters, accs, 'bo')
    plt.plot(combine, accs, 'go')
    plt.savefig("plotted")


if __name__ == '__main__':
    main("/hdd/PhD/nas/tas/parsecifar.txt")
