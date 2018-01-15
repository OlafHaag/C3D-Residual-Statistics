from __future__ import print_function
import sys
import csv
import warnings

import c3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.colors


def read_conditionals(filename):
    """
    Reads a C3D file and returns labels and conditionals on success.
    :param filename: C3D file to read.
    :return: marker labels and an array holding conditional values for marker x frames.
    :rtype: list, numpy.array or False
    """
    try:
        with open(filename, 'rb') as filehandle:
            print("Reading " + filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore UserWarning: missing parameter ANALOG:DESCRIPTIONS/LABELS
                reader = c3d.Reader(filehandle)
            nframes = reader.last_frame() - reader.first_frame() + 1
            print("Number of frames in the header info: {}".format(nframes))
            labels = reader.point_labels
            var_array = np.empty([len(labels), nframes])
            var_array.fill(np.NAN)
            for i, points, analog in reader.read_frames():
                if i > nframes:
                    break
                var_array[:, i-1] = points[:, 3]
        return labels, var_array
    except IOError:
        print("Error: File {} could not be opened!".format(filename))
        return False


def get_conditional_stats(var_array):
    """Takes array of conditional values frame x marker.
    Returns a list containing percentage information for conditionals on each label.
    :type var_array: numpy.array
    :rtype: numpy.array
    """
    num_markers = int(var_array.shape[0])
    res = np.empty((num_markers+1, 8), float)
    for i in range(num_markers):
        cond_eq_0 = 100.0 * np.sum(var_array[i, :] == 0) / var_array.shape[1]
        cond_le_1 = 100.0 * np.sum(np.logical_and(var_array[i, :] > 0, var_array[i, :] <= 1)) / var_array.shape[1]
        cond_le_2 = 100.0 * np.sum(np.logical_and(var_array[i, :] > 1, var_array[i, :] <= 2)) / var_array.shape[1]
        cond_le_5 = 100.0 * np.sum(np.logical_and(var_array[i, :] > 2, var_array[i, :] <= 5)) / var_array.shape[1]
        cond_le_10 = 100.0 * np.sum(np.logical_and(var_array[i, :] > 5, var_array[i, :] <= 10)) / var_array.shape[1]
        cond_le_30 = 100.0 * np.sum(np.logical_and(var_array[i, :] > 10, var_array[i, :] <= 30)) / var_array.shape[1]
        cond_g_30 = 100.0 * np.sum(var_array[i, :] > 30) / var_array.shape[1]
        missing = 100.0 * np.sum(var_array[i, :] < 0) / var_array.shape[1]
        res[i] = [cond_eq_0, cond_le_1, cond_le_2, cond_le_5, cond_le_10, cond_le_30, cond_g_30, missing]
    res[-1] = np.sum(res[:-1, :], axis=0)/num_markers  # Averages
    return res
    

def get_frame_stats(var_array):
    """Takes array of conditional values frame x marker.
    Returns a list containing percentage information for conditionals on each frame.
    :type var_array: numpy.array
    :rtype: numpy.array
    """
    num_frames = int(var_array.shape[1])
    res = np.empty((num_frames, 8), float)
    for i in range(num_frames):
        cond_eq_0 = 100.0 * np.sum(var_array[:, i] == 0) / var_array.shape[0]
        cond_le_1 = 100.0 * np.sum(np.logical_and(var_array[:, i] > 0, var_array[:, i] <= 1)) / var_array.shape[0]
        cond_le_2 = 100.0 * np.sum(np.logical_and(var_array[:, i] > 1, var_array[:, i] <= 2)) / var_array.shape[0]
        cond_le_5 = 100.0 * np.sum(np.logical_and(var_array[:, i] > 2, var_array[:, i] <= 5)) / var_array.shape[0]
        cond_le_10 = 100.0 * np.sum(np.logical_and(var_array[:, i] > 5, var_array[:, i] <= 10)) / var_array.shape[0]
        cond_le_30 = 100.0 * np.sum(np.logical_and(var_array[:, i] > 10, var_array[:, i] <= 30)) / var_array.shape[0]
        cond_g_30 = 100.0 * np.sum(var_array[:, i] > 30) / var_array.shape[0]
        missing = 100.0 * np.sum(var_array[:, i] < 0) / var_array.shape[0]
        res[i] = [cond_eq_0, cond_le_1, cond_le_2, cond_le_5, cond_le_10, cond_le_30, cond_g_30, missing]
    return res.T


def write_stats(labels, stats, **kwargs):
    """Writes statistics about conditionals to a csv file.
    Takes 'filename' as keyword argument.
    :param stats: 2-dimensional array holding marker label and categories in each row.
    :param kwargs: filename= CSV file to save statistics to.
    :type labels: list
    :type stats: numpy.array
    """
    filename = kwargs.pop('filename', None)
    if kwargs:
        raise TypeError("Unexpected **kwargs: %r" % kwargs)
    try:
        with open(filename, 'wb') as filehandle:
            print("Writing statistics to file {}".format(filename))
            csv_writer = csv.writer(filehandle)
            csv_writer.writerow(["Marker", "Cond == 0", "Cond <= 1", "Cond <= 2", "Cond <= 5", "Cond <= 10",
                                 "Cond <= 30", "Cond > 30", "Missing"])
            for i, label in enumerate(labels):
                # label, cond_eq_0, cond_le_1, cond_le_2, cond_le_5, cond_le_10, cond_le_30, cond_g_30, missing
                csv_writer.writerow([label] + [x for x in stats[i]])
    except TypeError:
        print("Error: No filename given to save CSV.")
    except (IOError, OSError):
        print("Error: Could not write to file {}".format(filename))


def print_stats(labels, stats):
    """Prints statistics about conditionals to stdout."""
    print("===================================================================================================")
    print("Label     |Cond == 0 |Cond <= 1 |Cond <= 2 |Cond <= 5 |Cond <= 10|Cond <= 30|Cond > 30 |Missing   |")
    for i, label in enumerate(labels):
        print("{}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|".format(label.ljust(10),
                                                                                                   *[x for x in stats[i]]))
    print("===================================================================================================")


def safe_ln(x, minval=0.0000000001):
    """Avoids division by Zero error in np.log(x)."""
    return np.log(x.clip(min=minval))


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
    
def plot_all_in_one(var_array, stats, filename=None):  # Stats could also be calculated inside from var_array
    """
    Analyses markers' conditional values by categorizing them and saving a visual graphic as <filename>.
    :param var_array: numpy array that holds conditional values for each frame per marker.
    :param filename: file path to save image to.
    """
    plt.clf()  # For multiprocessing to work, old plot has to go first.
    plt.close()
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Conditionals', size=20)
    grid = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[8, 8, 1])
    axes_frames = fig.add_subplot(grid[0, 0])
    axes_rel = fig.add_subplot(grid[1, 0], sharex=axes_frames)
    #plt.setp(axes_frames.get_xticklabels(), visible=False)
    axes_hist = fig.add_subplot(grid[:-1, 1])
    axes_cbar = fig.add_subplot(grid[2, :])

    levels = [-1.0, 0.0, 0.0001, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
    colors = ['black', 'palegreen', 'springgreen', 'mediumseagreen', 'seagreen', 'orange', 'red', 'firebrick', 'maroon']
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors[:-1])

    # marker x frame plot
    plt.setp(axes_frames, title='by frame')
    plt.setp(axes_frames, ylabel='Markers')
    plt.setp([axes_rel, axes_frames], xlabel='Frames')
    im = axes_frames.imshow(var_array, aspect=0.42*var_array.shape[1]/var_array.shape[0], cmap=cmap, norm=norm)  # FixMe: Interpolated has wrong colors
    
    # category x frame plot
    plt.setp(axes_rel, ylabel='Total %')
    data = get_frame_stats(var_array)
    frames = np.arange(0, var_array.shape[1])
    fill_colors = colors[:]
    fill_colors.append(fill_colors.pop(0))  # Shift colors by 1 to match stats indices
    # fixMe: doesn't look correct, too discrete y values
    axes_rel.fill_between(frames, 0, data[0], step='pre', color=fill_colors[0])
    for i in range(1, 7):
        axes_rel.fill_between(frames, np.sum(data[:i], axis=0), np.sum(data[:i+1], axis=0), step='pre', color=fill_colors[i])
    axes_rel.fill_between(frames, 100-data[-1], 100, step='pre', color=fill_colors[-1])
    
    axes_rel.set_ylim([0, 100])  # Always show range 0-100%
    axes_frames.set_xlim([0, var_array.shape[1]])  # Display whole frame range
    
    # pseudo histogram
    plt.setp(axes_hist, title='overall')
    values = np.roll(stats[-1], 1)  # Shift elements of averages, so missing is the first. Matches levels order.
    categories = ['-1', '0', '<01', '<02', '<05', '<10', '<30', '>30']  # Sort issue fixed in matplotlib master: no sort
    axes_hist.bar(categories, values, color=colors)
    axes_hist.set_ylim([0, 100])  # Always show range 0-100%
    plt.setp(axes_hist, xlabel='conditionals')
    plt.setp(axes_hist, ylabel='percent')
    
    # colorbar
    cbar = fig.colorbar(im, cax=axes_cbar, ticks=levels, orientation='horizontal')
    code_labels = ['-1\ninvalid', '0\ninterpolated', '<1', '<2', '<5\ngood', '<10\nbad', '<30', '>30']
    loc = cbar.get_ticks()
    loc += 0.5
    cbar.set_ticks([-0.5, 0.00005, 0.5, 1.5, 3.5, 7.5, 20.0, 65.0])
    cbar.set_ticklabels(code_labels)
    cbar.ax.set_title("color code")
    
    grid.tight_layout(fig, rect=[0, 0, 1, 0.95])
    if filename:
        print("Saving plot to {}".format(filename))
        plt.savefig(filename)
    else:
        plt.show()
    # plt.close()


def save_c3dstats(filename):
    """
    Reads a C3D file and saves the statistics as <filename>_stats.csv and <filename>_stats.png.
    :param filename: C3D file to analyze.
    :type filename: str
    :return: Success or Failure
    :rtype: bool
    """
    data = read_conditionals(filename)
    if data:
        labels = data[0]
        labels.append('Avg')
        var_array = data[1]
        stats = get_conditional_stats(var_array)
        write_stats(labels, stats, filename=filename.replace(".c3d", "_stats.csv"))
        plot_all_in_one(var_array, stats, filename=filename.replace(".c3d", "_stats.png"))
        return True
    else:
        return False
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = read_conditionals(sys.argv[1])
        if data:
            labels = data[0]
            labels.append('Avg')
            var_array = data[1]
            stats = get_conditional_stats(var_array)
            print_stats(labels, stats)
            plot_all_in_one(var_array, stats)
    else:
        print("Prints c3d file statistics on conditionals")
        print("Usage: c3dstats [filename]")

