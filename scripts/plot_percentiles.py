import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpm_plotter import plot_populations, read_from_terminal

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

"""plots time against values
    @param values list with values to plot that has the following dimensions:
    1st dimension: different timeseries to plot i.e. first is ABM timeseries, second is PDMM timeseries...
    2nd dimension: number of different outputs (percentiles) for one timeseries: first value is mean, second p05 ...
    3rd dimension: number of regions
    4th dimension: matrix with lines the number of timepoints and columns the compartments for that timepoint
"""
def plot_percentiles2(time, values, comp_to_plot, colors, region_names, time_series_labels, sum = False, y_label = "", save_dir=""):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    comps_names = ["Susceptible", "Exposed", "Carrier", "Infected", "Recovered", "Dead"]
    # iterate over all region
    for r in range(len(region_names)):
        mean_list = []
        # iterate over all model outputs e.g. ABM, PDMM, Hybrid
        fig, ax = plt.subplots()
        for s in range(len(values)):
            series = values[s]
            data = []
            # for one region iterate over all percentiles
            for p in range(len(series)):
                percentile = series[p]
                region_table = percentile[r]
                y = region_table[:, comp_to_plot[0]]
                label = comps_names[comp_to_plot[0]]
                if sum:
                    for c in range(1, len(comp_to_plot)):
                        y += region_table[:, comp_to_plot[c]]
                    label = y_label
                data.append(y)
                # case mean
                if p==0:
                    mean_list.append(y)
                    ax.plot(time, y, label = time_series_labels[s], color=colors[s])
                #case p25 or p75
                else:
                    ax.plot(time, y, color=colors[s], linestyle = "dotted", alpha=0.3)
                # fill between p25 and p75
            ax.fill_between(time, data[1], data[2], alpha=0.2, color=colors[s])
        ax.set_zorder(1)
        plt.ylabel(label)
        plt.xlabel("Time(days)")
        plt.subplots_adjust(bottom=0.15)
        plt.grid()
        plt.legend(bbox_to_anchor=(0.6, 1), loc="center left")
        # add MAPE
        for ts in range(1, len(time_series_labels)):
            MAPE = np.mean(np.abs(mean_list[0] - mean_list[ts])/mean_list[0])
            plt.figtext(0.58, 0.6 - (ts-1)*0.07, f'MAPE = {np.round(MAPE, 4)}', style='italic', color=colors[ts])
        fig.savefig(save_dir + "Percentiles_" + label + "_" + region_names[r]+".png")

def plot_percentiles(time, mean, percentiles, comp, compare = [], scaling_factor=1, label = [], filename=''
                     , region_names = ["Fürstenfeldbruck", "Dachau", "Starnberg", "München", "München Land", 
                                       "Freising", "Erding", "Ebersberg"], comp_name = "Infected agents"):
    for r in range(len(mean)):
        region_mean = mean[r] * scaling_factor
        region_p05 = percentiles[0][r] * scaling_factor
        region_p25 = percentiles[1][r] * scaling_factor
        region_p50 = percentiles[2][r] * scaling_factor
        region_p75 = percentiles[3][r] * scaling_factor
        region_p95 = percentiles[4][r] * scaling_factor
        fig = plt.figure()
        if(len(compare) > 0):
            for c in range(len(compare)):
                region_extrapolated = compare[c][r]
                plt.plot(time, region_extrapolated[:, comp], linestyle='--', label=label[c])
        plt.plot(time, region_p05[:, comp], label = 'p05', color='dimgray', linestyle="dotted")
        plt.plot(time, region_p25[:, comp], label = 'p25', color = 'dimgray', linestyle="dotted")
        plt.plot(time, region_p75[:, comp], label = 'p75', color='dimgray', linestyle="dotted")
        plt.plot(time, region_p95[:, comp], label = 'p95', color='dimgray', linestyle="dotted")
        #plt.plot(time, region_p50[:, comp], label = 'p50')
        plt.plot(time, region_mean[:, comp], label = 'mean', color='black')
        plt.fill_between(time, region_p05[:, comp], region_p95[:, comp], color='dimgray', alpha=0.2)
        plt.fill_between(time, region_p25[:, comp], region_p75[:, comp], color='dimgray', alpha=0.4)
        plt.legend()
        plt.xlabel("Time(days)")
        plt.ylabel(comp_name)
        plt.title(region_names[r])
        fig.savefig(filename + 'percentiles_' + str(r) +'_'+str(comp) + '.png')
        plt.close()

"""
    @param values_mean list with model output as first dimension, per model output list with mean output matrix per region
    @param values_percentiles list with model output as first dimension, and percentile list as second dimension, starting with p05 and ending with p95
"""
def plot_num_transitions(time, values_mean, percentile_values, comp, region_names, labels, colors=["blue", "green"], y_label="Number transitions total", title="all"):
    for r in range(len(region_names)):
        fig, ax = plt.subplots()
        for output in range(len(values_mean)):
            ax.plot(time, values_mean[output][r][:, comp], label = labels[output], color=colors[output])
        for output in range(len(percentile_values)):
            o_p05 = percentile_values[output][0]
            o_p25 = percentile_values[output][1]
            o_p75 = percentile_values[output][3]
            o_p95 = percentile_values[output][4]
            plt.fill_between(time, o_p05[r][:, comp], o_p95[r][:, comp], color=colors[output], alpha=0.1)
            plt.fill_between(time, o_p25[r][:, comp], o_p75[r][:, comp], color=colors[output], alpha=0.2)
            plt.plot(time, o_p05[r][:, comp], color=colors[output], linestyle="dotted")
            plt.plot(time, o_p25[r][:, comp], color=colors[output], linestyle="dotted")
            plt.plot(time, o_p75[r][:, comp], color=colors[output], linestyle="dotted")
            plt.plot(time, o_p95[r][:, comp], color=colors[output], linestyle="dotted")
        plt.ylabel(y_label)
        plt.xlabel("Time(days)")
        plt.legend()
        fig.savefig("Transitions_" + title + "_" + region_names[r]+".png")
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
    for l in range(len(flow_list)):
        for region in range(len(flow_list[l])):
            if scale == "local":
                flow_list[l][region] *= (100000.0/populations[region])
            elif scale == "global":
                flow_list[l][region] *= (100000.0/sum(populations))
            else:
                print("Unknown scale")
    return flow_list, real

def get_starting_populations(mean_list):
    populations = []
    for region in range(len(mean_list)):
        comps = mean_list[region][0,:]
        populations.append(np.sum(comps))
    return populations

def plot_flow(time, mean, percentiles, index, filename):
    for r in range(len(mean)):
        region_mean = mean[r][:, index] #* scaling_factor
        region_p05 = percentiles[0][r][:, index] #* scaling_factor
        region_p25 = percentiles[1][r][:, index] #* scaling_factor
        region_p50 = percentiles[2][r][:, index] #* scaling_factor
        region_p75 = percentiles[3][r][:, index] #* scaling_factor
        region_p95 = percentiles[4][r][:, index] #* scaling_factor
        fig = plt.figure()
        plt.plot(time, region_mean, label = 'mean')
        plt.plot(time, region_p05, label = 'p05', color='navy')
        plt.plot(time, region_p25, label = 'p25', color = 'dimgray')
        plt.plot(time, region_p50, label = 'p50')
        plt.plot(time, region_p75, label = 'p75', color='dimgray')
        plt.plot(time, region_p95, label = 'p95', color='navy')
        plt.fill_between(time, region_p05, region_p95, color='navy', alpha=0.2)
        plt.fill_between(time, region_p25, region_p75, color='dimgray', alpha=0.4)
        plt.legend()
        fig.savefig(filename + 'percentiles_' + str(r) + '.png')
        plt.close()

def plot_mean(time, mean, filename, labels, index_list):
    for region in range(len(mean)):
        fig = plt.figure()
        for i in index_list:
            plt.plot(time, mean[region][:, i], label = labels[i])
        plt.legend()
        fig.savefig(filename + str(region) + '.png')
        plt.close()

def add_compartments(result_list):
    acc_result_list = []
    for output in result_list:
        acc_output = []
        #output is a list with the result table for every region
        for region in output:
            acc_output.append(np.sum(region, axis=1).reshape(-1, 1))

        acc_result_list.append(acc_output)
    return acc_result_list

dir = "cpp/outputs/QuadWell/20241016_v1/"
save_dir = "scripts/Results/QuadWell/20241016_v1/"
#table_real, labels_real = read_from_terminal(dir + "output_extrapolated.txt")
#time = table_real[:,0]
num_regions = 4
num_comp = 6

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende


#list with mean output as first element and percentiles as following elements starting with p05 and ending with p95
ABM_list, time_ABM = read_mean_and_percentile_outputs(dir, "ABM", num_comp, num_regions)
PDMM_list, time_PDMM = read_mean_and_percentile_outputs(dir, "PDMM", num_comp, num_regions)
# ABM_list_from, ABM_time_from = read_mean_and_percentile_outputs(dir, "2.4_ABM_trans_from", num_comp, num_regions)
# ABM_list_to, ABM_time_to = read_mean_and_percentile_outputs(dir, "2.4_ABM_trans_to", num_comp, num_regions)
# PDMM_list_from, PDMM_time_from = read_mean_and_percentile_outputs(dir, "2.4_PDMM_trans_from", num_comp, num_regions)
# PDMM_list_to, PDMM_time_to = read_mean_and_percentile_outputs(dir, "2.4_PDMM_trans_to", num_comp, num_regions)

# ABM_list_from_acc = add_compartments(ABM_list_from)
# ABM_list_to_acc = add_compartments(ABM_list_to)
# PDMM_list_from_acc = add_compartments(PDMM_list_from)
# PDMM_list_to_acc = add_compartments(PDMM_list_to)

Hybrid_list, time_Hybrid = read_mean_and_percentile_outputs(dir, "Hybrid", num_comp, num_regions)

# get same lists summed up for all regions
ABM_list_accumulated = get_accumulated_output(ABM_list)
PDMM_list_accumulated = get_accumulated_output(PDMM_list)
Hybrid_list_accumulated = get_accumulated_output(Hybrid_list)

ABM_list_mean_p25_p75 = [ABM_list[0], ABM_list[2], ABM_list[4]]
ABM_list_acc_mean_p25_p75 = [ABM_list_accumulated[0], ABM_list_accumulated[2], ABM_list_accumulated[4]]

PDMM_list_mean_p25_p75 = [PDMM_list[0], PDMM_list[2], PDMM_list[4]]
PDMM_list_acc_mean_p25_p75 = [PDMM_list_accumulated[0], PDMM_list_accumulated[2], PDMM_list_accumulated[4]]

Hybrid_list_mean_p25_p75 = [Hybrid_list[0], Hybrid_list[2], Hybrid_list[4]]
Hybrid_list_acc_mean_p25_p75 = [Hybrid_list_accumulated[0], Hybrid_list_accumulated[2], Hybrid_list_accumulated[4]]

#[ABM_list_mean_p25_p75, PDMM_list_mean_p25_p75, Hybrid_list_mean_p25_p75]
# plot number infectious (C+I) for all three models and all regions
plot_percentiles2(time_ABM, [ABM_list_mean_p25_p75, PDMM_list_mean_p25_p75, Hybrid_list_mean_p25_p75], [2, 3], ["blue", "red", "green"], 
                  ["Focus_region", "Region_1", "Region_2", "Region_3"], 
                  ["ABM", "PDMM", "Spatial Hybrid"], sum=True, y_label="Number Infectious", save_dir=save_dir) # ["Fürstenfeldbruck", "Dachau", "Starnberg", "München", "München Land",  "Freising", "Erding", "Ebersberg"]
# plot number infectious (C+I) for all three models sum over all regions
plot_percentiles2(time_ABM, [ABM_list_acc_mean_p25_p75, PDMM_list_acc_mean_p25_p75, Hybrid_list_acc_mean_p25_p75], [2, 3], ["blue", "red", "green"], ["All_regions"],
                  ["ABM", "PDMM", "Spatial Hybrid"], sum=True, y_label="Number Infectious", save_dir=save_dir)

# plot_populations(time_Hybrid, Hybrid_list[0], [0, 1, 2, 3, 4, 5, 6, 7], "test")

# # plot number transitions for ABM and PDMM accumulated for all compartments
# plot_num_transitions(ABM_time_from, [ABM_list_from_acc[0], PDMM_list_from_acc[0]], [ABM_list_from_acc[1:]], comp=0, labels=["ABM", "PDMM"], region_names=["Focus region", "Region 1", "Region 2", "Region 3"])

# comps_names = ["Susceptible", "Exposed", "Carrier", "Infected", "Recovered", "Dead"]
# # plot number transitions for ABM and PDMM per compartment
# for c in range(len(comps_names)):
#     ylable = "Number transitions " + comps_names[c]
#     plot_num_transitions(ABM_time_from, [ABM_list_from[0], PDMM_list_from[0]], [ABM_list_from[1:]], comp=c, labels=["ABM", "PDMM"], region_names=["Focus region", "Region 1", "Region 2", "Region 3"], 
#                          y_label=ylable, title=comps_names[c])

#Infected
# plot_percentiles(time_ABM, ABM_list[0], ABM_list[1:], 3, filename='18Test_ABM_Infected_', compare=[PDMM_list[0]], 
#                  label=['PDMM mean', 'Hybrid mean'], region_names=["Focus region", "Region 1", "Region 2", "Region 3"])
# plot_percentiles(time_ABM, ABM_list_accumulated[0], ABM_list_accumulated[1:], 3, filename='18Test_ABM_infected_acc_',
#                  compare=[PDMM_list_accumulated[0]], label=['PDMM mean', 'Hybrid mean'], region_names=["All regions"])
