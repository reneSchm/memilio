import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from mpm_plotter import read_from_terminal

map_data = geopandas.read_file(os.path.join(
    os.getcwd(), '../tools/vg2500_12-31.gk3.shape/vg2500/VG2500_KRS.shp'))


output, time = read_from_terminal("../cpp/outputs/Munich/Vortrag_Martin/1Hybrid_comps_output_mean.txt")

map_data['Region'] = -1*np.ones(len(map_data))
map_data.Region = map_data.Region.astype('int')

map_data.at[238, 'Region'] = 4
map_data.at[228, 'Region'] = 1 
map_data.at[233, 'Region'] = 0 
map_data.at[242, 'Region'] = 2 
map_data.at[229, 'Region'] = 7 
map_data.at[231, 'Region'] = 6 
map_data.at[232, 'Region'] = 5 
map_data.at[223, 'Region'] = 3
map_data = map_data.loc[[238, 228, 233, 242, 229, 231, 232, 223]]
map_data = map_data[['ARS', 'Region', 'geometry']]

t=0
comp = 3

map_data['Values'] = -1*np.ones(len(map_data))
region_pops = np.zeros(8)
for r in range(region_pops.shape[0]):
    region_pops[r] = np.sum(output[0, 1 +(r*6):7 + (r*6)])

for r in range(len(map_data)):
    map_data.iat[r, 3] = (output[t, 1 + comp + map_data.Region.iloc[r]*6] / region_pops[map_data.Region.iloc[r]]) * 100000

fig, ax = plt.subplots()
ax.set_axis_off()
map_data.plot(map_data.Values, ax=ax, legend=False, vmin=0, vmax=4800)
plt.savefig('colored_map' + str(int(t)) + '.png', dpi=300)

print('')