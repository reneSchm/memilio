import pandas as pd
import matplotlib.pyplot as plt 

font_size = 16
plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

def plot_scaling_infected_transmissions(outputfile, resultfile):
    df_ABM = pd.read_csv(outputfile + "time_infected_transmissions_ABM.txt", sep=" ")
    df_PDMM = pd.read_csv(outputfile + "time_infected_transmissions_PDMM.txt", sep=" ")
    df_Hybrid = pd.read_csv(outputfile + "time_infected_transmissions_Hybrid.txt", sep=" ")

    fig, ax = plt.subplots()
    plt.grid()
    ax.scatter(df_ABM.sum_Infected, df_ABM.ABM_Time, label='ABM', color='indianred')
    ax.scatter(df_PDMM.sum_Infected, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    ax.scatter(df_Hybrid.sum_Infected, df_Hybrid.Hybrid_Time, label='Hybrid', color='green')
    ax.set_xlabel('Infected[#]')
    ax.set_ylabel('Time[s]')
    ax.legend()
    fig.savefig(resultfile + "time_infected.png")

    fig, ax = plt.subplots()
    plt.grid()
    ax.scatter(df_ABM.transmissions, df_ABM.ABM_Time, label='ABM', color='indianred')
    ax.scatter(df_PDMM.transmissions, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    ax.scatter(df_Hybrid.transmissions, df_Hybrid.Hybrid_Time, label='Hybrid', color='green')
    ax.set_xlabel('Transmissions[#]')
    ax.set_ylabel('Time[s]')
    ax.legend()
    fig.savefig(resultfile + "time_transmissions.png")

    fig, ax = plt.subplots()
    plt.grid()
    ax.scatter(df_ABM.deaths, df_ABM.ABM_Time, label='ABM', color='indianred')
    ax.scatter(df_PDMM.deaths, df_PDMM.PDMM_Time, label='PDMM', color='royalblue')
    ax.scatter(df_Hybrid.deaths, df_Hybrid.Hybrid_Time, label='Hybrid', color='green')
    ax.set_xlabel('Deaths[#]')
    ax.set_ylabel('Time[s]')
    ax.legend()
    fig.savefig(resultfile + "time_deaths.png")

def plot_scaling_susceptibles(outfile, resultfile):
    df = pd.read_csv(outfile + "time_sus_scaling.txt", sep=" ")
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.yscale("log")
    ax.scatter(df.na, df.ABM, label='ABM', color='indianred')
    ax.scatter(df.na, df.PDMM, label='PDMM', color='royalblue')
    ax.scatter(df.na, df.Hybrid, label='Hybrid', color='green')
    ax.set_xlabel('Agents[#]')
    ax.set_ylabel('Time[s]')
    ax.legend()
    fig.savefig(resultfile + "time_sus.png")

outputfile = 'cpp/outputs/QuadWell/time_measure/20241022_v2/'
resultfile = 'scripts/Results/QuadWell/time/20241022_v2/'

plot_scaling_infected_transmissions(outputfile, resultfile)
plot_scaling_susceptibles(outputfile, resultfile)
