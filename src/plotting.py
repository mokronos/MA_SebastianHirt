from matplotlib import gridspec
import matplotlib.pyplot as plt
import helpers
import os

reference_dir = 'data/raw/ReferenceSCIs'
compressed_dir = 'data/raw/DistortedSCIs'
labels_dir = 'labels/raw'
image_dir = 'images'

image_ext = '.bmp'
label_ext = '.txt'

# get filenames
reference_filenames = helpers.get_filenames(reference_dir)
compressed_filenames = helpers.get_filenames(compressed_dir)
labels_filenames = helpers.get_filenames(labels_dir)

reference_filenames.sort()
compressed_filenames.sort()
labels_filenames.sort()

# plot reference images in huge grid
# load images
# reference_filenames = reference_filenames[:6]
reference_images = []
for filename in reference_filenames:
    img = plt.imread(os.path.join(reference_dir, filename + image_ext))
    reference_images.append(img)

# plot images

gs = gridspec.GridSpec(5, 8, top=1., bottom=0., right=1., left=0., hspace=0.,
                       wspace=0.)

for g in gs:
    if len(reference_images) == 0:
        break
    ax = plt.subplot(g)
    ax.imshow(reference_images.pop(0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{reference_filenames.pop(0)}', fontsize=6)
#    ax.set_aspect('auto')

plt.savefig(os.path.join(image_dir, 'reference_images.pdf'), dpi=300)
# 35MB laggy
# plt.savefig(os.path.join(image_dir, 'reference_images_HD.pdf'), dpi=3000)
