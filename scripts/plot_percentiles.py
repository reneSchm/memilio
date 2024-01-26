import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpm_plotter import plot_populations

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
    
def plot(time, data1, data2, comp, labels=['data1', 'data2'], filename = 'plt', scaling_factor = 1,
          title = '', xlabel = '', ylabel = ''):
    for r in range(len(data1)):
        region_data1 = data1[r]
        region_data2 = data2[r] / scaling_factor
        fig, ax = plt.subplots()
        ax.plot(time, region_data1[:, comp], label = labels[0])
        ax.plot(time, region_data2[:, comp], label = labels[1])
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(filename + '_' + str(r) +'_'+str(comp) + '.png')
        plt.close()

def plot_percentiles(time, mean, percentiles, comp, extrapolated = [], scaling_factor=1, label = 'extrapolated', filename=''):
    for r in range(len(mean)):
        region_mean = mean[r] * scaling_factor
        region_p05 = percentiles[0][r] * scaling_factor
        region_p25 = percentiles[1][r] * scaling_factor
        region_p50 = percentiles[2][r] * scaling_factor
        region_p75 = percentiles[3][r] * scaling_factor
        region_p95 = percentiles[4][r] * scaling_factor
        fig = plt.figure()
        if(len(extrapolated) > 0):
            region_extrapolated = extrapolated[r]
            plt.plot(time, region_extrapolated[:, comp], linestyle='--', label=label)
        plt.plot(time, region_mean[:, comp], label = 'mean')
        plt.plot(time, region_p05[:, comp], label = 'p05', color='navy')
        plt.plot(time, region_p25[:, comp], label = 'p25', color = 'dimgray')
        plt.plot(time, region_p50[:, comp], label = 'p50')
        plt.plot(time, region_p75[:, comp], label = 'p75', color='dimgray')
        plt.plot(time, region_p95[:, comp], label = 'p95', color='navy')
        plt.fill_between(time, region_p05[:, comp], region_p95[:, comp], color='navy', alpha=0.2)
        plt.fill_between(time, region_p25[:, comp], region_p75[:, comp], color='dimgray', alpha=0.4)
        plt.legend()
        fig.savefig(filename + 'percentiles_' + str(r) +'_'+str(comp) + '.png')
        plt.close()

def read_mean_and_percentile_outputs(dir, prefix, num_comp, num_regions):
    #list contains subtables for mean as first element and subtables for 
    #percentiles as following elements starting with p05 and ending with p95
    read_list = []
    read_list.append(read_from_terminal(dir + prefix + "_output_mean.txt")[0])
    time = read_list[0][:, 0]
    read_list.append(read_from_terminal(dir + prefix + "_output_p05.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_output_p25.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_output_p50.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_output_p75.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_output_p95.txt")[0])
    subtable_list = []
    for table in read_list:
        subtable_list.append([table[:, i * num_comp + 1: (i + 1)*num_comp + 1] for i in range(num_regions)])
    return subtable_list, time

def get_accumulated_output(subtable_list):
    accumulated_list = []
    for i in range(len(subtable_list)):
        accumulated_list.append([sum(subtable_list[i])])
    return accumulated_list

def plot_percentiles_new_infections(time, mean, percentiles, real, scaling_factor, indices, factors, filename=''):
    for r in range(len(mean)):
        region_mean = mean[r] #* scaling_factor
        region_p05 = percentiles[0][r] #* scaling_factor
        region_p25 = percentiles[1][r] #* scaling_factor
        region_p50 = percentiles[2][r] #* scaling_factor
        region_p75 = percentiles[3][r] #* scaling_factor
        region_p95 = percentiles[4][r] #* scaling_factor
        fig = plt.figure()
        plt.scatter(time, real[:, r]/scaling_factor, marker="x", label="real", color='black')
        new_infections_mean = 0 * region_mean[:, 0]
        new_infections_p05 = 0 * region_p05[:, 0]
        new_infections_p25 = 0 * region_p25[:, 0]
        new_infections_p50 = 0 * region_p50[:, 0]
        new_infections_p75 = 0 * region_p75[:, 0]
        new_infections_p95 = 0 * region_p95[:, 0]
        for i in range(len(indices)):
            new_infections_mean += factors[i] * region_mean[:, i]
            new_infections_p05 += factors[i] * region_p05[:, i]
            new_infections_p25 += factors[i] * region_p25[:, i]
            new_infections_p50 += factors[i] * region_p50[:, i]
            new_infections_p75 += factors[i] * region_p75[:, i]
            new_infections_p95 += factors[i] * region_p95[:, i]
        plt.plot(time, new_infections_mean, label = 'mean')
        plt.plot(time, new_infections_p05, label = 'p05', color='navy')
        plt.plot(time, new_infections_p25, label = 'p25', color = 'dimgray')
        plt.plot(time, new_infections_p50, label = 'p50')
        plt.plot(time, new_infections_p75, label = 'p75', color='dimgray')
        plt.plot(time, new_infections_p95, label = 'p95', color='navy')
        plt.fill_between(time, new_infections_p05, new_infections_p95, color='navy', alpha=0.2)
        plt.fill_between(time, new_infections_p25, new_infections_p75, color='dimgray', alpha=0.4)
        plt.legend()
        fig.savefig(filename + 'percentiles_' + str(r) + '.png')
        plt.close()

def scale_new_infections(flow_list, real, populations, scale):
    population_real = [218579.0, 155449.0, 136747.0, 1487708.0, 349837.0, 181144.0, 139622.0, 144562.0]
    if scale == "local":
        for region in range(real.shape[1]):
            real[:, region] *= (100000.0/population_real[region])
    elif scale == "global":
        real *= (100000.0/sum(population_real))
    else:
        print("Unknown scale")
    print()

def get_starting_populations(mean_list):
    populations = []
    for region in range(len(mean_list)):
        comps = mean_list[region][0,:]
        populations.append(np.sum(comps))
    return populations
dir = "../cpp/outputs/Hybrid4/"
prefix =""
#table_real, labels_real = read_from_terminal(dir + "output_extrapolated.txt")
#time = table_real[:,0]
num_regions = 8
num_comp = 6

hybrid_comp_list, time_hybrid_comp = read_mean_and_percentile_outputs(dir, "comps", num_comp, num_regions)
hybrid_flow_list, time_hybrid_flows = read_mean_and_percentile_outputs(dir, "flows", num_comp, num_regions)

comp_list_accumulated = get_accumulated_output(hybrid_comp_list)
flow_list_accumulated = get_accumulated_output(hybrid_flow_list)

plot_populations(time_hybrid_comp, hybrid_comp_list[0], ["S", "E", "C", "I", "R", "D"]*len(hybrid_comp_list[0]), "cumulative")
plot_populations(time_hybrid_comp, comp_list_accumulated[0], ["S", "E", "C", "I", "R", "D"], "cumulative_acc")

real, time_real = read_from_terminal(dir+"new_infections.txt")
real = real[:, 1:]
real_accumulated = np.sum(real, axis=1).reshape((real.shape[0], 1))
# populations = get_starting_populations(hybrid_comp_list[0])
# scale_new_infections(hybrid_flow_list, )

plot_percentiles_new_infections(time_hybrid_flows, hybrid_flow_list[0], hybrid_flow_list[1:], real, scaling_factor=100, indices=[1, 2], factors=[0.1, 1], filename="flows_")
plot_percentiles_new_infections(time_hybrid_flows, flow_list_accumulated[0], flow_list_accumulated[1:], real_accumulated, scaling_factor=100, indices=[1, 2], factors=[0.1, 1], filename='flows_acc_')

plot_percentiles(time_hybrid_comp, hybrid_comp_list[0], hybrid_comp_list[1:], comp=3, scaling_factor=1, filename="comps_")
plot_percentiles(time_hybrid_comp, comp_list_accumulated[0], comp_list_accumulated[1:], comp=3, filename='comps_acc_', scaling_factor=1)

# ABM_list, time_ABM = read_mean_and_percentile_outputs(dir, "ABM", num_comp, num_regions)
# PDMM_list, time_PDMM = read_mean_and_percentile_outputs(dir, "PDMM", num_comp, num_regions)

# ABM_list_accumulated = get_accumulated_output(ABM_list)
# PDMM_list_accumulated = get_accumulated_output(PDMM_list)

# plot(time_ABM, ABM_list[0], PDMM_list[0], comp=3, labels=['ABM', 'PDMM'], title='Mean results of 10 runs', xlabel='Time', ylabel='Infected')
# plot(time_ABM, ABM_list_accumulated[0], PDMM_list_accumulated[0], comp=3, labels=['ABM', 'PDMM'], filename='acc', title='Mean results of 10 runs for all regions', xlabel='Time', ylabel='Infected')
#plot_percentiles(time_PDMM, PDMM_list[0], PDMM_list[1:], comp=3)
#plot_percentiles(time_PDMM, PDMM_list_accumulated[0], PDMM_list_accumulated[1:], comp=3, filename='accumulated')

#plot_percentiles(time_ABM, ABM_list[0], ABM_list[1:], comp=3, extrapolated=PDMM_list[0], label='PDMM mean')
#plot_percentiles(time_ABM, ABM_list_accumulated[0], ABM_list_accumulated[1:], comp=3, extrapolated=PDMM_list_accumulated[0], label='PDMM mean', filename='acc_')