#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:22:29 2022

@author: lifo
"""
import pandas as pd

import numpy as np

import seaborn as sns

import operator

import math

import networkx

from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

from matplotlib import pyplot as plt

# TODO: refactor methods

def pairwise_plot(
    df, baseline, margin=0.015, y_min=0, y_max=1, show_names_above=0.7,
    max_ncols=2, sns_context='talk', figsize=None, dpi=None, show_win_areas=False
):
    """
    Make pairwise plots using a dataframe with columns as a model performance
    and index as the dataset name.

    Parameters
    ----------
    df : DataFrame
        A dataframe containing the performance of models as columns, and the
        dataset names as index.
    baseline : str
        Name of the columns to use as baseline on the y-axis.
    margin : float, optional
        Margin separating the x=y line. If the difference between two models is 
        within +/- margin, this dataset will be considered as a draw.
        The default is 0.025.
    y_min : float, optional
        Minimum value on the axes. The default is 0.
    y_max : TYPE, optional
        Maximum value on the axes. The default is 1.
    show_names_above : float, optional
        If a difference between two models on a data is above this value,
        corresponding index name will be shown above the point.
        The default is 0.7.
    max_ncols : int, optional
        The maximum number of columns to draw. The number of rows is equal to 
        the number of columns in df - 1 divided by max_ncols, rounded up.
        The default is 2.
    sns_context : str, optional
        Type of seaborn context to apply, use None to not use sns style.
        The default is 'talk'.
    figsize : set, optional
        A two dimensional set as (x,y), with x the width of the figure and
        y its height. The default is None and will automatically adjust the 
        size of the figure.

    Returns
    -------
    fig : matplotlib.Figure
        The resulting pairwise plot.

    """
    if sns_context is not None:
        sns.set()
        sns.set_context(sns_context)
    competitors = df.columns.difference([baseline])
    if len(competitors)<max_ncols:
        ncols=len(competitors)
        nrows=1
    else:
        ncols = max_ncols
        nrows = int(np.ceil(len(competitors)/ncols))
    if figsize is None:
        figsize = (7.5*ncols, 7.5*nrows)
    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize, sharey=True, dpi=dpi
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    for i, comp in enumerate(competitors):
        if nrows > 1:
            limp = (y_max-y_min)*0.05
            ax[i // ncols, i % ncols].set_xlim(y_min-limp, y_max+limp)
            ax[i // ncols, i % ncols].set_ylim(y_min-limp, y_max+limp)
            ax[i // ncols, i %
                ncols].plot([y_min, y_max], [y_min, y_max], color='r')
            ax[i // ncols, i % ncols].plot([y_min, y_max-margin], [
                                           y_min + margin, y_max], color='orange', linestyle='--', alpha=0.75)
            ax[i // ncols, i % ncols].plot([y_min + margin, y_max], [
                                           y_min, y_max-margin], color='orange', linestyle='--', alpha=0.75)
            ax[i // ncols, i % ncols].annotate(
                '{}'.format(margin),
                (y_min + margin/2 + 0.0125, y_min+margin/2 - 0.01),
                fontsize=12
            )
            x = df[comp].values
            y = df[baseline].values
            ax[i // ncols, i % ncols].scatter(x, y, s=15, alpha=0.75, zorder=2)
            if np.any(np.abs(x-y)>=show_names_above):
                for j in range(df.shape[0]):
                    x1 = df.iloc[j][comp]
                    y1 = df.iloc[j][baseline]
                    if abs(x1 - y1) > show_names_above:
                        ax[i // ncols, i % ncols].annotate(
                            "{}".format(df.index[j]),
                            (x1, y1),
                            fontsize=14,
                            bbox=dict(boxstyle="round", alpha=0.1),
                        )
            if i % ncols == 0:
                ax[i // ncols, i % ncols].set_ylabel(baseline)
            ax[i // ncols, i % ncols].set_xlabel(comp)
    
            textstr = 'W - D - L\n{} - {} - {}'.format(sum(x+margin < y), sum(
                (x <= y+margin) & (y-margin <= x)), sum(x-margin > y))
            ax[i // ncols, i % ncols].text(0.05, 0.95, textstr, transform=ax[i // ncols, i % ncols].transAxes, fontsize=14,
                                           verticalalignment='top', bbox=props)
        if nrows == 1:
            if ncols==1:
            
                limp = (y_max-y_min)*0.05
                ax.set_xlim(y_min-limp, y_max+limp)
                ax.set_ylim(y_min-limp, y_max+limp)
                ax.plot([y_min, y_max], [y_min, y_max], color='r')
                ax.plot([y_min, y_max-margin], [
                                               y_min + margin, y_max], color='orange', linestyle='--', alpha=0.75)
                ax.plot([y_min + margin, y_max], [
                                               y_min, y_max-margin], color='orange', linestyle='--', alpha=0.75)
                
                ax.annotate(
                    '{}'.format(margin),
                    (y_min + margin/2 + 0.0125, y_min+margin/2 - 0.01),
                    fontsize=12
                )
                x = df[comp].values
                y = df[baseline].values
                ax.scatter(x, y, s=15, alpha=0.75, zorder=2)
                if np.any(np.abs(x-y)>=show_names_above):
                    for j in range(df.shape[0]):
                        x1 = df.iloc[j][comp]
                        y1 = df.iloc[j][baseline]
                        if abs(x1 - y1) > show_names_above:
                            ax.annotate(
                                "{}".format(df.index[j]),
                                (x1, y1),
                                fontsize=14,
                                bbox=dict(boxstyle="round", alpha=0.1)
                            )
                if i % ncols == 0:
                    ax.set_ylabel(baseline)
                ax.set_xlabel(comp)
    
                textstr = 'W - D - L\n{} - {} - {}'.format(sum(x+margin < y), sum(
                    (x <= y+margin) & (y-margin <= x)), sum(x-margin > y))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                                               verticalalignment='top', bbox=props)
            else:
                limp = (y_max-y_min)*0.05
                ax[i % ncols].set_xlim(y_min-limp, y_max+limp)
                ax[i % ncols].set_ylim(y_min-limp, y_max+limp)
                ax[i %
                    ncols].plot([y_min, y_max], [y_min, y_max], color='r')
                ax[i % ncols].plot([y_min, y_max-margin], [
                                               y_min + margin, y_max], color='orange', linestyle='--', alpha=0.75)
                ax[i % ncols].plot([y_min + margin, y_max], [
                                               y_min, y_max-margin], color='orange', linestyle='--', alpha=0.75)
                
                ax[i % ncols].annotate(
                    '{}'.format(margin),
                    (y_min + margin/2 + 0.0125, y_min+margin/2 - 0.01),
                    fontsize=12
                )
                x = df[comp].values
                y = df[baseline].values
                ax[i % ncols].scatter(x, y, s=15, alpha=0.75, zorder=2)
                if np.any(np.abs(x-y)>=show_names_above):
                    for j in range(df.shape[0]):
                        x1 = df.iloc[j][comp]
                        y1 = df.iloc[j][baseline]
                        if abs(x1 - y1) > show_names_above:
                            ax[i % ncols].annotate(
                                "{}".format(df.index[j]),
                                (x1, y1),
                                fontsize=14,
                                bbox=dict(boxstyle="round", alpha=0.1)
                            )
                if i % ncols == 0:
                    ax[i % ncols].set_ylabel(baseline)
                ax[i % ncols].set_xlabel(comp)
        
                textstr = 'W - D - L\n{} - {} - {}'.format(sum(x+margin < y), sum(
                    (x <= y+margin) & (y-margin <= x)), sum(x-margin > y))
                ax[i % ncols].text(0.05, 0.95, textstr, transform=ax[i % ncols].transAxes, fontsize=14,
                                               verticalalignment='top', bbox=props)
    return fig

def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, width=10, labels=False, path=None, highlight=None):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = _wilcoxon_holm(df_perf=df_perf, alpha=alpha)
    if p_values is not None:
        _graph_ranks(average_ranks.values, average_ranks.keys(), p_values, 
                    cd=None, reverse=True, width=width, textspace=1.25, labels=labels,
                    highlight=highlight)

        font = {'family': 'sans-serif',
                'color': 'black',
                'weight': 'normal',
                'size': 18,
                }
        if title:
            plt.title(title, fontdict=font, y=0.9, x=0.5)
        if path is None:
            plt.savefig('cd-diagram.png', bbox_inches='tight')

def _form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)

def _wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    print(friedman_p_value)

    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(
            df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy'], dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i]
                           [1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(
        m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(
        classifiers), columns=np.unique(sorted_df_perf['dataset_name']))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(
        axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets



def _graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None, highlight=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    Needs matplotlib to work.
    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.
    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional): if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    width = float(width)
    textspace = float(textspace)
    
    def lloc(_list, n):
         """
         List location in list of list structure.
         Enable the use of negative locations:
         -1 is the last element, -2 second last...
         """
         if n < 0:
             return len(_list[0]) + n
         return n
     
    def nth(_list, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(_list, n)
        return [a[n] for a in _list]

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
        """
        if len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height*1.05))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(_list):
        return [a * hf for a in _list]

    def wfl(_list):
        return [a * wf for a in _list]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        if nnames[i] == highlight:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=linewidth, color='red')
        else:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075,
                 format(ssums[i], '.4f'), ha="right", va="center", size=10)
        if nnames[i] == highlight:
            text(textspace - 0.2, chei,
                 filter_names(nnames[i]), ha="right", va="center", size=18, color='red')
        else:
            text(textspace - 0.2, chei,
                 filter_names(nnames[i]), ha="right", va="center", size=18)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        if nnames[i] == highlight:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=linewidth, color='red')
        else:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075,
                 format(ssums[i], '.4f'), ha="left", va="center", size=10)
        if nnames[i] == highlight:
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
                 ha="left", va="center", size=18, color='red')
        else:
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
                 ha="left", va="center", size=18)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = _form_cliques(p_values, nnames)
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height