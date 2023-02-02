import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'src')
from utils import ensure_dir

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
                    image_borders=None,
                    image_size=(2.5, 2.5),
                    row_labels=None,
                    figure_title=None,
                    font_size=12,
                    subplot_padding=None,
                    save_path=None,
                    show_figure=True):
    """
    Display rows of images

    Arg(s):
        images : list[list[np.array]]
            2D array of images to display
            images can be in format of C x H x W or H x W x C
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
    if image_borders is not None:
        assert len(image_borders) == n_rows
        assert len(image_borders[0]) == n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(image_size[0] * n_cols, image_size[1] * n_rows))

    for row in range(n_rows):
        for col in range(n_cols):
            # Obtain correct axis
            if n_rows == 1 and n_cols == 1:
                ax = axs
            elif n_rows == 1:
                ax = axs[col]
            else:
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

            # Change border color
            if image_borders is not None:
                plt.setp(ax.spines.values(), color=image_borders[row][col], linewidth=2.0)


            # Remove tick marks
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

    # Set figure title
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=font_size)

    # Pad if number is provided
    if subplot_padding is not None:
        plt.tight_layout(pad=subplot_padding)
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show_figure:
        plt.show()

    return fig, axs

def bar_graph(data,
              labels=None,
              groups=None,
              title=None,
              ylabel=None,
              xlabel_rotation=0,
              save_path=None,
              show_plot=True):
    '''
    Given data, make a bar graph

    Arg(s):
        data : N x C np.array
            N : number of data points
            C : number of bar classes
        labels : list[str]
            C length list of labels for each bar
        groups : list[str]
            N list of group names
        title : str
            title for bar graph
        ylabel : str
            label for y-axis
        xlabel_rotation : int
            how much to rotate x labels by if they overlap
        save_path : str
            if not None, the path to save bar graph to
    '''
    fig, ax = plt.subplots()
    assert len(data.shape) == 2, "Expected 2D data, received {}D data.".format(len(data.shape))
    n_groups, n_classes = data.shape

    # Parameters for bar graphs
    x_pos = np.arange(n_classes)
    width = 1 / n_groups
    if labels is None:
        labels = ["" for i in range(n_classes)]
    if groups is None:
        groups = [i for i in range(n_groups)]

    mid_idx = n_groups // 2
    if n_groups % 2 == 0: # Even number of groups
        for group_idx, group_data in enumerate(data):
            if group_idx < mid_idx:
                ax.bar(x_pos - width * ((mid_idx - group_idx) * 2 - 1) / 2,
                       group_data,
                       alpha=0.75,
                       ecolor='black',
                       capsize=10,
                       label=groups[group_idx],
                       width=width)
            else:
                ax.bar(x_pos + width * ((group_idx - mid_idx) * 2 + 1) / 2,
                       group_data,
                       alpha=0.75,
                       ecolor='black',
                       capsize=10,
                       label=groups[group_idx],
                       width=width)

    else:  # Odd number of groups
        for group_idx, group_data in enumerate(data):
            if group_idx < mid_idx:
                ax.bar(x_pos - 1 / 2 + width * group_idx,
                    group_data,
                    alpha=0.75,
                    ecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)
            elif group_idx == mid_idx:
                ax.bar(x_pos - width / 2,
                    group_data,
                    alpha=0.75,
                    ecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)
            else:
                ax.bar(x_pos - width / 2 + (group_idx - mid_idx) * width,
                    group_data,
                    alpha=0.75,
                    ecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)

    # Set prettiness
    ax.set_xticks(x_pos, labels)
    plt.setp(ax.get_xticklabels(), rotation=xlabel_rotation)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    # If save_path is not None, save graph
    if save_path is not None:
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    # Show figure
    if show_plot:
        plt.show()
    plt.close()

def histogram(data,
              n_bins=10,
              labels=None,
              data_range=None,
              color=None,
              title=None,
              xlabel=None,
              ylabel=None,
              marker=None,
              save_path=None,
              show=True):
    '''
    Plot histogram of data provided

    Arg(s):
        data : np.array or sequence of np.array
            Data for histogram
    '''
    plt.hist(data,
             bins=n_bins,
             label=labels,
             range=data_range,
             color=color)

    # Marker is a vertical line marking original
    if marker is not None:
        plt.axvline(x=marker, color='r')

    # Make legend
    if labels is not None:
        plt.legend()
    # Set title and axes labels
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()

# move to visualizations.py
def plot(xs,
         ys,
         labels=None,
         title=None,
         xlabel=None,
         ylabel=None,
         xlimits=None,
         ylimits=None,
         scatter=True,
         line=True,
         highlight=None,
         highlight_label=None,
         save_path=None,
         show=False):
    '''

    Arg(s):
        xs : list[list[float]]
            x values
        ys : list[list[float]]
            y values
        title : str
            title of plot
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        xlimits : [float, float] or None
            limits for x-axis
        ylimits : [float, float] or None
            limits for y-axis
        scatter : bool or list[bool]
            denoting if should show each data point or not
        line : bool or list[bool]
            denoting if should connect lines or not
        highlight : (list[float], list[float])
            tuple of data point(s) to accentuate
        save_path : str
            path to save graph to
        show : bool
            whether or not to display graph

    '''
    plt.clf()
    n_lines = len(xs)
    if labels is None:
        labels = [None for i in range(n_lines)]

    assert len(ys) == n_lines, "ys list must be same length as xs. Received {} and {}".format(len(ys), n_lines)
    assert len(labels) == n_lines, "Labels list must be same length as xs. Received {} and {}".format(len(labels), n_lines)

    # Determine plot types
    if type(scatter) == bool:
        scatter = [scatter for i in range(n_lines)]
    else:
        len(scatter) == n_lines, "scatter list must be same length as xs. Received {} and {}".format(len(scatter), n_lines)
    if type(line) == bool:
        line = [line for i in range(n_lines)]
    else:
        len(line) == n_lines, "line list must be same length as xs. Received {} and {}".format(len(line), n_lines)
    # Highlight certain point
    if highlight is not None:
        highlight_x, highlight_y = highlight
        format_str = 'ys'
        marker_size = 10
        zorder = 3
        if highlight_label is not None:
            plt.plot(
                highlight_x,
                highlight_y,
                format_str,
                markersize=marker_size,
                zorder=zorder,
                label=highlight_label)
        else:
            plt.plot(
                highlight_x,
                highlight_y,
                format_str,
                markersize=marker_size,
                zorder=zorder)

    # Plot lines
    for idx in range(n_lines):
        x = xs[idx]
        y = ys[idx]
        label = labels[idx]

        format_str = 'o'
        if scatter[idx] and line[idx]:
            format_str = '-o'
        elif not scatter[idx] and line[idx]:
            format_str = '-'

        if label is not None:
            plt.plot(x, y, format_str, zorder=1, label=label)
        else:
            plt.plot(x, y, format_str, zorder=1)

    # Add limits to axes
    if xlimits is not None:
        plt.xlim(xlimits)
    if ylimits is not None:
        plt.ylim(ylimits)

    # Set title and labels
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if labels[0] is not None:
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.clf()