"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





def cm_ranking(sc, ths=[512, 1024, 2048, 4096]):
    ls = sc.shape[0]
    colors = ["red", "yellow", "lime", "cyan", "blue"]
    out = ["gray"] * ls
    for i in range(ls):
        for c, th in zip(colors[: len(ths) + 1], ths + [ls]):
            if sc[i] < th:
                out[i] = c
                break
    sid = np.argsort(sc)[::-1]
    out = np.array(out)[sid]
    return out, sid

def cm_RdBl(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)

def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)

def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 1.0, 0, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out






def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    num = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * num

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * num
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, all_axes = plt.subplots( 1, num, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if num == 1:
        all_axes = [all_axes]
    for i, (img, ax) in enumerate(zip(imgs, all_axes)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)
    return fig



def plot_image_grid(
    imgs,
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    fig=None,
    adaptive=True,
    figs=2.0,
    return_fig=False,
    set_lim=False,
):
    """Plot a grid of images.
    Args:
        imgs: a list of lists of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    num_rows, num_cols = len(imgs), len(imgs[0])
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * num_cols

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs[0]]  # W / H
    else:
        ratios = [4 / 3] * num_cols

    figsize = [sum(ratios) * figs, num_rows * figs]
    if fig is None:
        fig, all_axes = plt.subplots(
            num_rows, num_cols, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    else:
        all_axes = fig.subplots(num_rows, num_cols, gridspec_kw={"width_ratios": ratios})
        fig.figure.set_size_inches(figsize)
    if num_rows == 1:
        all_axes = [all_axes]

    for j in range(num_rows):
        for i in range(num_cols):
            ax = all_axes[j][i]
            ax.imshow(imgs[j][i], cmap=plt.get_cmap(cmaps[i]))
            ax.set_axis_off()
            if set_lim:
                ax.set_xlim([0, imgs[j][i].shape[1]])
                ax.set_ylim([imgs[j][i].shape[0], 0])
            if titles:
                ax.set_title(titles[j][i])
    if isinstance(fig, plt.Figure):
        fig.tight_layout(pad=pad)
    if return_fig:
        return fig, all_axes
    else:
        return all_axes



def plot_keypoints(kpts, colors="lime", point_size=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        point_size: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=point_size, linewidths=0, alpha=alpha)




def plot_matches(kpts0, kpts1, color=None, line_width=1.5, point_size=4, alpha=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        line_width: width of the lines.
        point_size: size of the end points (no endpoint if point_size=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = sns.color_palette("husl", n_colors=len(kpts0))
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if line_width > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(xyA=(kpts0[i, 0], kpts0[i, 1]), xyB=(kpts1[i, 0], kpts1[i, 1]), 
                                                      coordsA=ax0.transData, coordsB=ax1.transData, 
                                                      axesA=ax0, axesB=ax1, zorder=1, 
                                                      color=color[i], linewidth=line_width, clip_on=True, 
                                                      alpha=alpha, label=None if labels is None else labels[i], picker=5.0,)
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if point_size > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=point_size,
            label=None if labels is None or len(labels) == 0 else labels[0],)
        ax1.scatter(
            kpts1[:, 0], kpts1[:, 1], c=color, s=point_size,
            label=None if labels is None or len(labels) == 0 else labels[1],)



def plot_matches2(kpts0, kpts1, color=None, line_width=1.5, point_size=4, alpha=1.0, labels=None, 
                  axes=None, captions=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        line_width: width of the lines.
        point_size: size of the end points (no endpoint if point_size=0)
        labels: labels for each match.
        axes: matplotlib axes to plot on.
        captions: list of strings for captions to display (e.g., ['AspanFormer', 'Matches: 97', ...]).
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        if labels is not None:
            # Define colors for true and false matches
            color = ['#32CD32' if label else 'red' for label in labels]
        else:
            color = sns.color_palette("husl", n_colors=len(kpts0))
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if line_width > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=line_width,
                clip_on=True,
                alpha=alpha,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # Freeze the axes to prevent the transform from changing
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if point_size > 0:
        ax0.scatter(
            kpts0[:, 0],
            kpts0[:, 1],
            c=color,
            s=point_size,
        )
        ax1.scatter(
            kpts1[:, 0],
            kpts1[:, 1],
            c=color,
            s=point_size,
        )
    if captions is not None:
        formatted_captions = [f"{caption:.1f}" if isinstance(caption, float) else
                               str(caption) for caption in captions]
        caption_text = '\n'.join(formatted_captions)
        ax0.text(
            0.01, 0.99, caption_text, 
            transform=ax0.transAxes, 
            fontsize=6, 
            verticalalignment='top', 
            horizontalalignment='left', 
            color='white', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
            zorder=3  # Ensure captions are in front of all other elements
        )







def plot_matches3(kpts0, kpts1, color=None, line_width=1.5, point_size=4, alpha=1.0, labels=None, 
                  axes=None, captions=None):
    """Plot matches for a pair of existing images with captions in front of all other elements.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        line_width: width of the lines.
        point_size: size of the end points (no endpoint if point_size=0)
        labels: labels for each match.
        axes: matplotlib axes to plot on.
        captions: list of strings for captions to display (e.g., ['AspanFormer', 'Matches: 97', ...]).
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        if labels is not None:
            # Define colors for true and false matches
            color = ['#32CD32' if label else 'red' for label in labels]
        else:
            color = sns.color_palette("husl", n_colors=len(kpts0))
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    # Plot captions first to ensure they are not covered by lines or points
    if captions is not None:
        formatted_captions = [f"{caption:.1f}" if isinstance(caption, float) else
                               str(caption) for caption in captions]
        caption_text = '\n'.join(formatted_captions)
        ax0.text(
            0.01, 0.99, caption_text, 
            transform=ax0.transAxes, 
            fontsize=6, 
            verticalalignment='top', 
            horizontalalignment='left', 
            color='white', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
            zorder=5  # Ensure captions are in front of all other elements
        )

    if line_width > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=line_width,
                clip_on=True,
                alpha=alpha,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # Freeze the axes to prevent the transform from changing
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if point_size > 0:
        ax0.scatter(
            kpts0[:, 0],
            kpts0[:, 1],
            c=color,
            s=point_size,
            zorder=2
        )
        ax1.scatter(
            kpts1[:, 0],
            kpts1[:, 1],
            c=color,
            s=point_size,
            zorder=2
        )











if __name__=="__main__":
    values = np.linspace(0, 1, 256)

    # Apply the custom colormap to these values
    colors = cm_RdGn(values)

    # Visualize the colormap
    plt.figure(figsize=(8, 3))
    plt.imshow([colors], aspect='auto')
    plt.title("Custom Red to Green Colormap")
    plt.axis('off')
    # plt.show()
    plt.imsave('colormap.png', [colors], format='png')







def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
    axes=None,
    **kwargs,
):
    if axes is None:
        axes = plt.gcf().axes

    ax = axes[idx]
    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
        **kwargs,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )
    return t


def draw_epipolar_line(line, axis, imshape=None, color="b", label=None, alpha=1.0, visible=True):
    if imshape is not None:
        h, w = imshape[:2]
    else:
        _, w = axis.get_xlim()
        h, _ = axis.get_ylim()
        imshape = (h + 0.5, w + 0.5)
    # Intersect line with lines representing image borders.
    X1 = np.cross(line, [1, 0, -1])
    X1 = X1[:2] / X1[2]
    X2 = np.cross(line, [1, 0, -w])
    X2 = X2[:2] / X2[2]
    X3 = np.cross(line, [0, 1, -1])
    X3 = X3[:2] / X3[2]
    X4 = np.cross(line, [0, 1, -h])
    X4 = X4[:2] / X4[2]

    # Find intersections which are not outside the image,
    # which will therefore be on the image border.
    Xs = [X1, X2, X3, X4]
    Ps = []
    for p in range(4):
        X = Xs[p]
        if (0 <= X[0] <= (w + 1e-6)) and (0 <= X[1] <= (h + 1e-6)):
            Ps.append(X)
            if len(Ps) == 2:
                break

    # Plot line, if it's visible in the image.
    if len(Ps) == 2:
        art = axis.plot(
            [Ps[0][0], Ps[1][0]],
            [Ps[0][1], Ps[1][1]],
            color,
            linestyle="dashed",
            label=label,
            alpha=alpha,
            visible=visible,
        )[0]
        return art
    else:
        return None


def get_line(F, kp):
    hom_kp = np.array([list(kp) + [1.0]]).transpose()
    return np.dot(F, hom_kp)


def plot_epipolar_lines(points0, points1, F, color="b", axes=None, labels=None, a=1.0, visible=True):
    if axes is None:
        axes = plt.gcf().axes
    assert len(axes) == 2

    for ax, kps in zip(axes, [points1, points0]):
        _, w = ax.get_xlim()
        h, _ = ax.get_ylim()

        imshape = (h + 0.5, w + 0.5)
        for i in range(kps.shape[0]):
            if ax == axes[0]:
                line = get_line(F.transpose(0, 1), kps[i])[:, 0]
            else:
                line = get_line(F, kps[i])[:, 0]
            draw_epipolar_line(
                line,
                ax,
                imshape,
                color=color,
                label=None if labels is None else labels[i],
                alpha=a,
                visible=visible,
            )








def plot_heatmaps(heatmaps, vmin=0.0, vmax=None, cmap="Spectral", a=0.5, axes=None):
    if axes is None:
        axes = plt.gcf().axes
    artists = []
    for i in range(len(axes)):
        a_ = a if isinstance(a, float) else a[i]
        art = axes[i].imshow(
            heatmaps[i],
            alpha=(heatmaps[i] > vmin).float() * a_,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        artists.append(art)
    return artists







def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def plot_cumulative(
    errors: dict,
    thresholds: list,
    colors=None,
    title="",
    unit="-",
    logx=False,
):
    thresholds = np.linspace(min(thresholds), max(thresholds), 100)

    plt.figure(figsize=[5, 8])
    for method in errors:
        recall = []
        errs = np.array(errors[method])
        for th in thresholds:
            recall.append(np.mean(errs <= th))
        plt.plot(
            thresholds,
            np.array(recall) * 100,
            label=method,
            c=colors[method] if colors else None,
            linewidth=3,
        )

    plt.grid()
    plt.xlabel(unit, fontsize=25)
    if logx:
        plt.semilogx()
    plt.ylim([0, 100])
    plt.yticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.ylabel(title + "Recall [%]", rotation=0, fontsize=25)
    plt.gca().yaxis.set_label_coords(x=0.45, y=1.02)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.yticks(rotation=0)

    plt.legend(
        bbox_to_anchor=(0.45, -0.12),
        ncol=2,
        loc="upper center",
        fontsize=20,
        handlelength=3,
    )
    plt.tight_layout()

    return plt.gcf()






