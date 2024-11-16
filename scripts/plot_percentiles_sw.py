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
def plot_percentiles2(time, values, comp_to_plot, colors, region_names, time_series_labels, sum = False, 
                      y_label = "", save_dir="", error="MAPE", figsize = (11, 9)):
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    comps_names = ["Susceptible", "Exposed", "Carrier", "Infected", "Recovered", "Dead"]
    # iterate over all region
    for r in range(len(region_names)):
        mean_list = []
        # iterate over all model outputs e.g. ABM, PDMM, Hybrid
        fig, ax = plt.subplots(figsize = figsize)
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
            ax.fill_between(time, data[1], data[2], alpha=0.2, color="darkgray")
        ax.set_zorder(1)
        plt.ylabel(label)
        plt.xlabel("Time(days)")
        plt.subplots_adjust(bottom=0.15)
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, 1.08), loc="center", ncol = 2)
        # add MAPE
        for ts in range(1, len(time_series_labels)):
            err = -1
            if(error == "MAPE"):
                err = np.mean(np.abs(mean_list[0] - mean_list[ts])/mean_list[0])
            elif(error == "MAE"):
                err = np.mean(np.abs(mean_list[0] - mean_list[ts]))
            elif(error == "MSE"):
                err = np.mean((mean_list[0] - mean_list[ts])**2)
            if(error == "All"):
                MAPE = np.mean(np.abs(mean_list[0] - mean_list[ts])/mean_list[0])
                MAE = np.mean(np.abs(mean_list[0] - mean_list[ts]))
                MSE = np.mean((mean_list[0] - mean_list[ts])**2)
                plt.figtext(0.15, 0.82 - (ts-1)*0.05, f'MAPE = {np.round(MAPE, 4)}', style='italic', color=colors[ts])
                plt.figtext(0.15, 0.62 - (ts-1)*0.05, f'MAE = {np.round(MAE, 4)}', style='italic', color=colors[ts])
                plt.figtext(0.15, 0.42 - (ts-1)*0.05, f'MSE = {np.round(MSE, 3)}', style='italic', color=colors[ts])
            else:
                plt.figtext(0.58, 0.6 - (ts-1)*0.07, f'{error} = {np.round(err, 4)}', style='italic', color=colors[ts])
        plt.tight_layout()
        fig.subplots_adjust(top=0.89)
        fig.savefig(save_dir + label + "_"+ error +".png")

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
    read_list.append(read_from_terminal(dir + prefix + "_mean.txt")[0])
    time = read_list[0][:, 0]
    read_list.append(read_from_terminal(dir + prefix + "_p05.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_p25.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_p50.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_p75.txt")[0])
    read_list.append(read_from_terminal(dir + prefix + "_p95.txt")[0])
    subtable_list = []
    for table in read_list:
        subtable_list.append([table[:, i * num_comp + 1: (i + 1)*num_comp + 1] for i in range(num_regions)])
    return subtable_list, time

def get_accumulated_output(subtable_list):
    accumulated_list = []
    for i in range(len(subtable_list)):
        accumulated_list.append([sum(subtable_list[i])])
    return accumulated_list

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

dir = "cpp/outputs/20241031_sw/"
save_dir = "scripts/Results/20241031_sw/"
#table_real, labels_real = read_from_terminal(dir + "output_extrapolated.txt")
#time = table_real[:,0]
num_regions = 1
num_comp = 6

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende


#list with mean output as first element and percentiles as following elements starting with p05 and ending with p95
ABM_list, time_ABM = read_mean_and_percentile_outputs(dir, "abm_20000_combined", num_comp, num_regions)
PDMM_list, time_PDMM = read_mean_and_percentile_outputs(dir, "pdmm_0_combined", num_comp, num_regions)
Hybrid_list_2, time_Hybrid_2 = read_mean_and_percentile_outputs(dir, "hybrid_2_combined", num_comp, num_regions)
Hybrid_list_5, time_Hybrid_5 = read_mean_and_percentile_outputs(dir, "hybrid_5_combined", num_comp, num_regions)
Hybrid_list_10, time_Hybrid_10 = read_mean_and_percentile_outputs(dir, "hybrid_10_survival", num_comp, num_regions)

# get same lists summed up for all regions
ABM_list_accumulated = get_accumulated_output(ABM_list)
PDMM_list_accumulated = get_accumulated_output(PDMM_list)
Hybrid_list_2_accumulated = get_accumulated_output(Hybrid_list_2)
Hybrid_list_5_accumulated = get_accumulated_output(Hybrid_list_5)

ABM_list_mean_p25_p75 = [ABM_list[0], ABM_list[2], ABM_list[4]]
ABM_list_acc_mean_p25_p75 = [ABM_list_accumulated[0], ABM_list_accumulated[2], ABM_list_accumulated[4]]

PDMM_list_mean_p25_p75 = [PDMM_list[0], PDMM_list[2], PDMM_list[4]]
PDMM_list_acc_mean_p25_p75 = [PDMM_list_accumulated[0], PDMM_list_accumulated[2], PDMM_list_accumulated[4]]

Hybrid_list_2_mean_p25_p75 = [Hybrid_list_2[0], Hybrid_list_2[2], Hybrid_list_2[4]]
Hybrid_list_2_acc_mean_p25_p75 = [Hybrid_list_2_accumulated[0], Hybrid_list_2_accumulated[2], Hybrid_list_2_accumulated[4]]

Hybrid_list_5_mean_p25_p75 = [Hybrid_list_5[0], Hybrid_list_5[2], Hybrid_list_5[4]]
Hybrid_list_5_acc_mean_p25_p75 = [Hybrid_list_5_accumulated[0], Hybrid_list_5_accumulated[2], Hybrid_list_5_accumulated[4]]

Hybrid_list_10_mean_p25_p75 = [Hybrid_list_10[0], Hybrid_list_10[2], Hybrid_list_10[4]]

#[ABM_list_mean_p25_p75, PDMM_list_mean_p25_p75, Hybrid_list_mean_p25_p75]
# plot number infectious (C+I) for all three models and all regions
plot_percentiles2(time_ABM, [ABM_list_mean_p25_p75, PDMM_list_mean_p25_p75, Hybrid_list_2_mean_p25_p75, Hybrid_list_5_mean_p25_p75], 
                  [3], ["tab:blue", "tab:orange", "tab:green", "tab:red"], ["Region_0"], ["ABM", "PDMM", "Temporal Hybrid (2)", "Temporal Hybrid (5)"], 
                  sum=True, y_label="Infected Compartment", save_dir=save_dir, error="All") 
