import numpy as np
import matplotlib.pyplot as plt

def read_from_terminal(filename):
    with open(filename) as file:
        lines = file.readlines()
        skip_to = 0
        labels = ""
        # find first line of the output table
        for i in range(len(lines)):
            if "Time" in lines[i]:
                skip_to = i + 1
                # read labels
                labels = lines[i].split()[1:]
                break
        # throw error if table was not found
        if labels == "":
            raise EOFError("Could not find results table in " + filename)
        result = []
        for i in range(skip_to, len(lines)):
            result+=[[]]
            for txt in lines[i].split():
                try:
                    result[-1] += [float(txt)]
                except ValueError:
                    # remove entries from failed line (should be empty)
                    result = result[:-1]
                    return np.array(result), labels
            if len(result[-1]) != len(labels) + 1:
                # remove entries from failed line
                result = result[:-1]
                return np.array(result), labels
        return np.array(result), labels

def plot_populations(time, metapopulations, labels, name):
    fig_ctr = 1
    lbl_ctr = 0
    for x in metapopulations:
        ys = [np.zeros_like(time)]
        for y in reversed([x[:, i] for i in range(len(x[0]))]):
            ys = [y + ys[0]] + ys

        plt.figure()
        
        for i in range(len(ys) - 1):
            plt.fill_between(time, ys[i], ys[i+1], label=labels[lbl_ctr])
            plt.plot(time, ys[i], c='black')
            lbl_ctr += 1
        plt.plot(time, ys[-1], c='black')
        
        plt.legend()
        plt.savefig(name+str(fig_ctr - 1)+".png")        
        fig_ctr += 1

if __name__ == "__main__":
    n_comps = 3
    n_regions = 4
    table, labels = read_from_terminal("output.txt")
    time = table[:,0]
    subtables = [table[:, i*n_comps+1:(i+1)*n_comps+1] for i in range(n_regions)]
    # subtables = [table[:, 1 + i * 6: 2+i*6] for i in range(8)]
    plot_populations(time, subtables, labels, "mpm")

    # subtables = [table[:, 1:4],table[:, 4:7]]
    # subtables = [table[:, i*6+1: (i+1)*6+1] for i in range(8)]
    # plot_populations(time, subtables, labels, "mpm")

    # plot_populations(time, [np.sum(subtables, axis=(0,2))[:,None]], labels, "mpm_total")

    # total = np.sum(subtables, axis=(0,2))
    # plt.plot(range(len(total)), total)
    # plt.savefig("total_mpm.png")

    # # print(np.sum(table[1:,1:], axis=0))
    # # print(np.diff(np.sum(table[1:,1:], axis=0)))
    # # print(np.average(table[1:,1:], axis=0))
    # # plt.plot(time[1:], table[1:,2], label="out")
    # # plt.plot(time[1:], table[1:,1], label="in")
    # diff = np.diff(table[1:,1:], axis=1)
    # # plt.plot(time[1:], [-np.sum(diff[:i+1]) for i in range(len(diff))], label="running diff")
    # plt.plot(time[2:], [np.average(table[1:i+1,1:], axis=0) for i in range(1,len(diff))], label="running avg")
    # # print(np.sum(diff))
    # plt.legend()
    # plt.savefig("agents_in_out.png")

    # tab = table[1:,1:]
    # tab_foc = tab[:,3*6:4*6]
    # print(np.sum(tab) - np.sum(tab_foc))