import numpy as np
import matplotlib.pyplot as plt

def show_image(image):
    '''
    Given np.array image, display using matplotlib
    '''
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

def show_image_rows(images,
                    image_labels=None,
                    image_size=(2.5, 2.5),
                    row_labels=None,
                    figure_title=None,
                    font_size=12,
                    save_path=None):
    """
    Display rows of images

    Arg(s):
        images : list[list[np.array]]
            2D array of images to display
        image_labels : list[list[str]] or None
            2D array of image labels, must be same shape as iamges
        image_size : (float, float)
            width, height of each image
        row_labels : list[str]
            list of labels for each row, must be same length as len(images)
        figure_title : str
            title for overall figure
        font_size : int
            font size
        save_path : str
            path to save figure to
    """

    n_rows, n_cols = len(images), len(images[0])
    # Shape sanity checks
    if image_labels is not None:
        assert len(image_labels) == n_rows
        assert len(image_labels[0]) == n_cols
    if row_labels is not None:
        assert len(row_labels) == n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(image_size[0] * n_cols, image_size[1] * n_rows))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axs[row, col]

            image = images[row][col]


def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5),
                   title=None, tlist=None, filename=None):
    from robustness.tools.vis_tools import get_axis

    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)

            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()