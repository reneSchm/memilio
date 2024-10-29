import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter 

font_size = 16
plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

def plot_scaling_infected_transmissions(outputfile, resultfile, figsize=(8, 6)):
    df_ABM = pd.read_csv(outputfile + "time_infected_transmissions_ABM.txt", sep=" ")
    df_PDMM = pd.read_csv(outputfile + "time_infected_transmissions_PDMM.txt", sep=" ")
    df_Hybrid = pd.read_csv(outputfile + "time_infected_transmissions_Hybrid.txt", sep=" ")

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax.grid(alpha=0.5)
    l1 = ax.scatter(df_ABM.sum_Infected, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = ax2.scatter(df_PDMM.sum_Infected, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = ax2.scatter(df_Hybrid.sum_Infected, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    ax.set_xlabel('Infected[#]')
    ax.set_ylabel('ABM Time[s]')
    ax2.set_ylabel('PDMM/Hybrid Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    plt.legend(l, labs , loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_infected.png")

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax.grid(alpha=0.5)
    l1 = ax.scatter(df_ABM.transmissions, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = ax2.scatter(df_PDMM.transmissions, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = ax2.scatter(df_Hybrid.transmissions, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    ax.set_xlabel('Transmissions[#]')
    ax.set_ylabel('ABM Time[s]')
    ax2.set_ylabel('PDMM/Hybrid Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    plt.legend(l, labs , loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_transmissions.png")

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax.grid(alpha=0.5)
    l1 = ax.scatter(df_ABM.deaths, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = ax2.scatter(df_PDMM.deaths, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = ax2.scatter(df_Hybrid.deaths, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    ax.set_xlabel('Deaths[#]')
    ax.set_ylabel('ABM Time[s]')
    ax2.set_ylabel('PDMM/Hybrid Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    plt.legend(l, labs , loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_deaths.png")

def plot_scaling_susceptibles(outfile, resultfile, xticks=[1400, 2600, 5000, 10000, 20000, 40000]):
    df = pd.read_csv(outfile + "time_sus_scaling.txt", sep=" ")
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.grid(alpha=0.5)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.scatter(df.na, df.ABM, label='ABM', color='indianred')
    ax.scatter(df.na, df.PDMM, label='PDMM', color='royalblue')
    ax.scatter(df.na, df.Hybrid, label='Spatial Hybrid', color='green')
    ax.set_xlabel('Agents[#]')
    ax.set_ylabel('Time[s]')
    ax.legend()
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.savefig(resultfile + "time_sus.png")

def plot_scaling_infected_transmissions_v1(outputfile, resultfile, figsize=(20, 6)):
    df_ABM = pd.read_csv(outputfile + "time_infected_transmissions_ABM.txt", sep=" ")
    df_PDMM = pd.read_csv(outputfile + "time_infected_transmissions_PDMM.txt", sep=" ")
    df_Hybrid = pd.read_csv(outputfile + "time_infected_transmissions_Hybrid.txt", sep=" ")

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)
    axs[0].grid(alpha=0.5)
    axs[1].grid(alpha=0.5)
    axs[2].grid(alpha=0.5)
    l1 = axs[0].scatter(df_ABM.sum_Infected, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = axs[0].scatter(df_PDMM.sum_Infected, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = axs[0].scatter(df_Hybrid.sum_Infected, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[2].scatter(df_PDMM.sum_Infected, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    axs[1].scatter(df_Hybrid.sum_Infected, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[1].set_xlabel('Infected[#]')
    axs[0].set_ylabel('Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.legend(l, labs, bbox_to_anchor=(0.667,1.0), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_infected_v1.png")

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)
    axs[0].grid(alpha=0.5)
    axs[1].grid(alpha=0.5)
    axs[2].grid(alpha=0.5)
    l1 = axs[0].scatter(df_ABM.transmissions, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = axs[0].scatter(df_PDMM.transmissions, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = axs[0].scatter(df_Hybrid.transmissions, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[2].scatter(df_PDMM.transmissions, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    axs[1].scatter(df_Hybrid.transmissions, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[1].set_xlabel('Transmissions[#]')
    axs[0].set_ylabel('Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.legend(l, labs, bbox_to_anchor=(0.667,1.0), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_transmissions_v1.png")

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)
    axs[0].grid(alpha=0.5)
    axs[1].grid(alpha=0.5)
    axs[2].grid(alpha=0.5)
    l1 = axs[0].scatter(df_ABM.deaths, df_ABM.ABM_Time, label='ABM', color='indianred')
    l2 = axs[0].scatter(df_PDMM.deaths, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    l3 = axs[0].scatter(df_Hybrid.deaths, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[2].scatter(df_PDMM.deaths, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    axs[1].scatter(df_Hybrid.deaths, df_Hybrid.Hybrid_Time, label='Spatial Hybrid', color='green')
    axs[1].set_xlabel('Deaths[#]')
    axs[0].set_ylabel('Time[s]')
    l = [l1, l2, l3]
    labs = [li.get_label() for li in l]
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.legend(l, labs, bbox_to_anchor=(0.667,1.0), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(resultfile + "time_deaths_v1.png")

outputfile = 'cpp/outputs/time_QW/'
resultfile = 'scripts/Results/time_QW/'

plot_scaling_infected_transmissions_v1(outputfile, resultfile)
plot_scaling_infected_transmissions(outputfile, resultfile)
#plot_scaling_susceptibles(outputfile, resultfile)
