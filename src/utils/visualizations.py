import numpy as np
import matplotlib.pyplot as plt

def show_image(image, title=None, save_path=None):
    '''
    Given np.array image, display using matplotlib
    '''
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    # Remove tick marks
    plt.tick_params(bottom=False, left=False)

    # Add title
    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def make_grid(flattened, items_per_row):
    '''
    Given a 1D list of items and how many elements per row, return a 2D list. Last row padded with None
    Helper to be called before show_image_rows()

    Arg(s):
        flattened : list[any]
            1D list of anything
        items_per_row : int
            number of elements per row

    Returns:
        list[list[any]] : 2D list of elements in a grid
    '''
    length = len(flattened)
    grid = []
    for i in range(0, length, items_per_row):
        if i + items_per_row <= length:
            grid.append(flattened[i: i + items_per_row])
        else:
            padded_row = flattened[i:]
            while len(padded_row) < items_per_row:
                padded_row.append(None)
            grid.append(padded_row)
    return grid


def show_image_rows(images,
                    image_titles=None,
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
        image_titles : list[list[str]] or None
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
    if image_titles is not None:
        assert len(image_titles) == n_rows
        assert len(image_titles[0]) == n_cols
    if row_labels is not None:
        assert len(row_labels) == n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(image_size[0] * n_cols, image_size[1] * n_rows))

    for row in range(n_rows):
        for col in range(n_cols):
            # Obtain correct axis
            ax = axs[row, col]

            # Display the image
            image = images[row][col]
            # For padding
            if image is None:
                continue

            # Matplotlib expects RGB channel to be in the back
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

            ax.imshow(image)

            # Display row text if first image in row
            if row_labels is not None and col == 0:
                ax.set_ylabel(row_labels[row], fontsize=font_size)
            # Display image title
            if image_titles is not None:
                ax.set_title(image_titles[row][col], fontsize=font_size)

            # Remove tick marks
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

    # Set figure title
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=font_size)

    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()