# standalone python scrip - generates a pgm image used by map_reader.cpp
import os
import geopandas
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon

# download and extract the data into the tools path below. Download link:
# https://daten.gdz.bkg.bund.de/produkte/vg/vg2500/aktuell/vg2500_12-31.gk3.shape.zip
map_data = geopandas.read_file(os.path.join(
    os.getcwd(), 'tools/vg2500_12-31.gk3.shape/vg2500/VG2500_LAN.shp'))
    # '../memilio/tools/vg2500_12-31.gk3.shape/vg2500/VG2500_KRS.shp'))
geometries = map_data.geometry

# remove ocean areas
geometries = geometries.loc[:15]
#geometries = geometries.loc[[238, 228, 233, 242, 229, 231, 232, 223]]
# 238             München
# 228              Dachau
# 233    Fürstenfeldbruck
# 242           Starnberg
# 229           Ebersberg
# 231              Erding
# 232            Freising
# 223             München

# remove islands from germany etc (i.e. keep only state boundaries)
for i in range(geometries.shape[0]):
    row = geometries.loc[i]
    if type(row) is not Polygon:
        geometries.loc[i] = list(row.geoms)[-1]
    # else:
    #     geometries.loc[i] = geometries.loc[i]
    # plt.plot(*polygon.exterior.coords.xy, label=map_data["GEN"].iloc[i])
# plt.legend()
# plt.savefig("loc.png")

color_range = 255

for i in [3]:
    dpi = 100 * i
    geometries.plot(color="white", edgecolor="black", linewidth=0.5/dpi, aspect = 1) # results in greyscale plot with light thin lines
    plt.axis('off')
    plt.tight_layout()
    png_name="geom_dpi="+str(dpi)+".png"
    img_name="potential_dpi="+str(dpi)+".pgm"
    plt.savefig(png_name, dpi=dpi)

    image = plt.imread(png_name)[:,:,0] # pick only red color channel (image is greyscale)
    c = int((image.shape[1] - image.shape[0]) / 2)
    image = np.where(image[:,c:-c] < 1.0, 0.0, 1.0) # crop image, set colors to either 0 or 1
    
    plt.imsave("potential_dpi="+str(dpi)+".png", image)
    with open(img_name, 'w') as file:
        file.write("P2\n")
        file.write(str(image.shape[1]) + " " + str(image.shape[0]) + "\n")
        file.write(str(color_range) + "\n")
        np.savetxt(file, np.array(image * color_range, dtype=int), fmt="%i")
