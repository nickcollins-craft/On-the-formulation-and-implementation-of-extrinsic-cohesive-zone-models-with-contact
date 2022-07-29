import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Declare the name of the folder where the images of the Rhombus hole sample are
# kept (these can be downloaded from the associated Zenodo repository)
image_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the folder where the image will be saved
save_folder = "/path/to/your/folder/goes/here/"

# Plot time!
width_in_inches = (8.27-2*1.5/2.54)/1.25
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width_in_inches, 0.625*width_in_inches))

# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

mesh_image = mpimg.imread(image_folder + 'clipped_rhombus_wireframe.png')
end_image = mpimg.imread(image_folder + 'clipped_rhombus.png')

# Do the raw mesh
ax0 = plt.subplot(121)
plt.imshow(mesh_image)
fig.patch.set_visible(False)
ax0.axis('off')

# Do the end plot
ax1 = plt.subplot(122)
plt.imshow(end_image)
# Do the colour bar, specifying colour scheme, range, label, and size
cmap = matplotlib.cm.plasma
norm = matplotlib.colors.Normalize(vmin=0.12, vmax=0.45)
ax1.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label='Total displacement (mm)')

plt.tight_layout()

# Save the figure using the dpi command (this helps it render the mesh properly
# internally before it saves it as a .pdf)
plt.savefig(save_folder + "Figure_15.pdf", dpi=320)

plt.show()
