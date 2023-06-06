from matplotlib import gridspec
import matplotlib.pyplot as plt
import helpers
from config import PATHS

ids = list(range(1, 41))
ref_paths = helpers.create_paths(PATHS["images_scid_ref"], ids)

save_path = "./images/"

gs = gridspec.GridSpec(5, 8, top=1., bottom=0., right=1., left=0., hspace=0.,
                       wspace=0.)

for idx, g in enumerate(gs):
    if len(ref_paths) == 0:
        break
    ax = plt.subplot(g)
    ax.imshow(plt.imread(ref_paths.pop(0)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Image: {idx+1}', fontsize=6)


plt.savefig(save_path + "reference_images.pdf", dpi=300)
plt.savefig(save_path + "reference_images.png", dpi=300)
# 35MB laggy
plt.savefig(save_path + "reference_images_HD.pdf", dpi=3000)
plt.savefig(save_path + "reference_images_HD.png", dpi=3000)
