from __future__ import print_function
import os
import sys
import csv
import warnings
import argparse

import c3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.colors

from . import version


def read_conditionals(filename):
    """
    Reads a C3D file and returns labels and conditionals on success.
    :param filename: C3D file to read.
    :return: marker labels and an array holding conditional values for marker x frames.
    :rtype: tuple|bool
    """
    try:
        with open(filename, 'rb') as file_handle:
            print("Reading " + filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore UserWarning: missing parameter ANALOG:DESCRIPTIONS/LABELS
                reader = c3d.Reader(file_handle)
            n_frames = reader.last_frame() - reader.first_frame() + 1
            print("Number of frames in the header info: {}".format(n_frames))
            labels = reader.point_labels
            conditionals = np.empty([len(labels), n_frames])
            conditionals.fill(np.NAN)
            for i, points, analog in reader.read_frames():
                if i > n_frames:
                    break
                conditionals[:, i-1] = points[:, 3]
        return labels, conditionals
    except IOError:
        print("Error: File {} could not be opened!".format(filename))
        return False


def check_conditionals_input(conditionals):
    try:
        shape = conditionals.shape
    except AttributeError:
        raise ValueError("Conditionals input must be a numpy array!")
    if len(shape) != 2:
        raise ValueError("Conditionals input must be a 2D numpy array!")
    return True


# Todo: use pandas instead of numpy
def get_conditional_stats(conditionals, thresholds=None):
    """Takes array of conditional values (frame x marker).
    Returns a list containing percentage information for conditionals on each point label.
    :type conditionals: numpy.ndarray
    :type thresholds: list
    :rtype: numpy.ndarray
    """
    sane_input = check_conditionals_input(conditionals)
    num_markers = int(conditionals.shape[0])

    if not thresholds:
        thresholds = [0, 1, 2, 5, 10, 30]
        
    res = np.empty((num_markers+1, 8), float)
    for i in range(num_markers):
        cond_less_first = np.sum(conditionals[i, :] < thresholds[0])
        cond_equal_first = np.sum(conditionals[i, :] == thresholds[0])
        cond_inner = [np.sum(np.logical_and(conditionals[i, :] > thresholds[j], conditionals[i, :] <= thresholds[j+1]))
                      for j in range(len(thresholds))[:-1]]
        cond_greater_last = np.sum(conditionals[i, :] > thresholds[-1])
        res[i] = [cond_equal_first] + cond_inner + [cond_greater_last, cond_less_first]
        res[i] = res[i] * 100.0 / conditionals.shape[1]
    res[-1] = np.sum(res[:-1, :], axis=0)/num_markers  # Averages
    return res


def get_frame_stats(conditionals, thresholds=None):
    """Takes array of conditional values (frame x marker).
    Returns a list containing percentage information for conditionals on each frame.
    :type conditionals: numpy.array
    :type thresholds: list
    :rtype: numpy.array
    """
    sane_input = check_conditionals_input(conditionals)
    num_frames = int(conditionals.shape[1])
    if not thresholds:
        thresholds = [0, 1, 2, 5, 10, 30]
        
    res = np.empty((num_frames, 8), float)
    for i in range(num_frames):
        cond_less_first = np.sum(conditionals[:, i] < thresholds[0])
        cond_equal_first = np.sum(conditionals[:, i] == thresholds[0])
        cond_inner = [
            np.sum(np.logical_and(conditionals[:, i] > thresholds[j], conditionals[:, i] <= thresholds[j + 1]))
            for j in range(len(thresholds))[:-1]]
        cond_greater_last = np.sum(conditionals[:, i] > thresholds[-1])
        res[i] = [cond_equal_first] + cond_inner + [cond_greater_last, cond_less_first]
        res[i] = res[i] * 100.0 / conditionals.shape[0]
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
    # Todo: use pandas for more flexibility in thresholds.
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
    
    
def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="""Output statistics about residual values in C3D motion capture files.""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-v", "--ver", action='version', version='%(prog)s {}'.format(version))
    parser.add_argument("input.c3d", nargs='*', type=str, help="C3D files to analyze.")
    parser.add_argument("-s", "--save", action='store_true', help="Save statistics to files.")
    
    args = vars(parser.parse_args(argv))
    src_filepaths = args['input.c3d']
    do_save = args['save']
    
    res = list()
    for c3d_file in src_filepaths:
        if do_save:
            res.append(save_c3dstats(c3d_file))
        else:
            data = read_conditionals(c3d_file)
            if data:
                labels = data[0]
                labels.append('Avg')
                conditionals = data[1]
                stats = get_conditional_stats(conditionals)
                print_stats(labels, stats)
                plot_all_in_one(conditionals, stats)
    
    num_errors = len(res) - sum(res)
    if num_errors > 0:
        print("ERROR: {} files could not be processed.".format(num_errors))
    return False if num_errors else True


if __name__ == "__main__":
    exit_code = int(not main())
    sys.exit(exit_code)
