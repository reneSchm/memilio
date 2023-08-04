import os
import geopandas
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon

im = np.reshape(np.loadtxt("map.pgm", skiprows=4), (480, 480))

for i in range(480):
    for j in range(480):
        if 0 < im[i,j] < 255:
            print(i / 480, j/480)

exit()

map_data = geopandas.read_file(os.path.join(
    os.getcwd(), 'tools/vg2500_12-31.gk3.shape/vg2500/VG2500_LAN.shp'))
geometries = map_data.geometry

# remove ocean areas
geometries = geometries.loc[:15]

# remove islands etc (i.e. keep only state boundaries)
for i in range(geometries.shape[0]):
    row = geometries.loc[i]
    if type(row) is not Polygon:
        geometries.loc[i] = list(row.geoms)[-1]
    # else:
    #     geometries.loc[i] = geometries.loc[i]
    # plt.plot(*polygon.exterior.coords.xy, label=map_data["GEN"].iloc[i])
# plt.legend()
# plt.savefig("loc.png")

def bucket_fill(x, index, color, replace=None, queue=None):
    queue = [tuple(index)]
    if replace is None:
        replace = x[index]
    if np.all(replace == color):
        return
    while len(queue) != 0:
        index = queue[0]
        queue = queue[1:]
        if np.all(x[index] == replace):
            x[index] = color
            if 0 < index[0]:
                queue += [(index[0] - 1, index[1])]
            if index[0] + 1 < x.shape[0]:
                queue += [(index[0] + 1, index[1])]
            if 0 < index[1]:
                queue += [(index[0], index[1] - 1)]
            if index[1] + 1 < x.shape[1]:
                queue += [(index[0], index[1] + 1)]

color_range = 255

for i in [1, 3, 10]:
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
    
    # bucket_fill(image, [0, 0], 0) # fill outside of map
    plt.imsave("potential_dpi="+str(dpi)+".png", image)
    with open(img_name, 'w') as file:
        file.write("P2\n")
        file.write(str(image.shape[1]) + " " + str(image.shape[0]) + "\n")
        file.write(str(color_range) + "\n")
        np.savetxt(file, np.array(image * color_range, dtype=int), fmt="%i")

