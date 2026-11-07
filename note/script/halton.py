"""Visual demonstrations of the Halton low-discrepancy sequence.

Four experiments, each with its own plotting function:

1. ``plot_stratification_base2`` -- in one dimension, base 2, the first ``2**m``
   points place *exactly one* point in every dyadic interval, at every
   resolution ``m``.
2. ``plot_coprime_effect`` -- in two dimensions, bases that share a factor
   collapse onto a lattice / the diagonal, while coprime bases stay
   equidistributed.
3. ``plot_scrambling_effect`` -- Owen scrambling randomises the point set: the
   fixed origin and the correlated diagonal stripes of a higher base pair
   disappear, while the coverage (cell counts) stays the same.
4. ``plot_stripe_mechanism`` -- why the stripes exist: the leading digits of the
   two coordinates advance in lockstep with ``n``, so points line up on segments
   of slope ``b_x / b_y``; large bases make the second digit too small to smear
   those lines.  ``plot_stripe_base_comparison`` draws the same budget for three
   base pairs so the effect can be compared directly.

The Halton machinery is implemented here from scratch with NumPy so the script
stays self-contained and fully typed.  ``curobo`` uses
``scipy.stats.qmc.Halton(..., scramble=True)``, which scrambles with one
permutation per digit position; this script uses the stronger nested
(per-prefix) Owen variant described in ``note/halton.md``.

Usage::

    python note/script/halton.py                  # show the figures
    python note/script/halton.py --save-dir out   # write PNGs instead
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Halton sequence
# ---------------------------------------------------------------------------


def digits_needed(max_index: int, base: int) -> int:
    """Return how many base-``base`` digits are needed to write ``max_index``."""
    if max_index < 0:
        raise ValueError("max_index must be non-negative")
    if base < 2:
        raise ValueError("base must be at least 2")
    n_digits = 1
    capacity = base
    while capacity <= max_index:
        capacity *= base
        n_digits += 1
    return n_digits


def base_digits(indices: IntArray, base: int, n_digits: int) -> IntArray:
    """Write ``indices`` in base ``base``: shape ``(len(indices), n_digits)``.

    Digits are stored least-significant first, i.e. column ``k`` holds the
    coefficient of ``base**k``.
    """
    digits: IntArray = np.empty((indices.shape[0], n_digits), dtype=np.int64)
    work = indices.copy()
    for k in range(n_digits):
        digits[:, k] = work % base
        work //= base
    return digits


def digits_to_unit(digits: IntArray, base: int) -> FloatArray:
    """Map base-``base`` digits to ``[0, 1)`` by reversing them about the point."""
    weights: FloatArray = np.float64(base) ** -np.arange(1, digits.shape[1] + 1)
    values: FloatArray = digits.astype(np.float64) @ weights
    return values


def radical_inverse(indices: IntArray, base: int) -> FloatArray:
    """Return ``phi_base(n)`` for every ``n`` in ``indices``."""
    max_index = int(indices.max()) if indices.size > 0 else 0
    return digits_to_unit(
        base_digits(indices, base, digits_needed(max_index, base)), base
    )


def van_der_corput(count: int, base: int) -> FloatArray:
    """Return the first ``count`` terms of the base-``base`` van der Corput sequence."""
    if count < 1:
        raise ValueError("count must be positive")
    return radical_inverse(np.arange(count, dtype=np.int64), base)


def _permutation_table(base: int, level: int, rng: np.random.Generator) -> IntArray:
    """Draw one independent digit permutation per prefix of length ``level``."""
    n_prefixes = base**level
    table: IntArray = np.empty((n_prefixes, base), dtype=np.int64)
    for row in range(n_prefixes):
        table[row] = rng.permutation(base)
    return table


def owen_scramble_digits(
    digits: IntArray, base: int, rng: np.random.Generator
) -> IntArray:
    """Apply nested uniform (Owen) scrambling to a digit matrix.

    Digit ``k`` is permuted with a permutation that depends on the
    already-scrambled prefix ``a_0 ... a_{k-1}``, so every node of the digit
    tree gets its own random relabelling of its children.
    """
    n_points, n_digits = digits.shape
    scrambled: IntArray = np.empty((n_points, n_digits), dtype=np.int64)
    prefix: IntArray = np.zeros(n_points, dtype=np.int64)
    for level in range(n_digits):
        table = _permutation_table(base, level, rng)
        scrambled[:, level] = table[prefix, digits[:, level]]
        prefix = prefix * base + scrambled[:, level]
    return scrambled


def halton_points(
    count: int,
    bases: Sequence[int],
    *,
    scramble: bool = False,
    seed: int = DEFAULT_SEED,
) -> FloatArray:
    """Return ``count`` points of the (optionally scrambled) Halton sequence.

    Dimension ``j`` uses ``bases[j]``; each dimension is scrambled with its own
    independent random stream derived from ``seed``.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if len(bases) == 0:
        raise ValueError("at least one base is required")

    indices = np.arange(count, dtype=np.int64)
    points: FloatArray = np.empty((count, len(bases)), dtype=np.float64)
    for dim, base in enumerate(bases):
        n_digits = digits_needed(count - 1, base)
        digits = base_digits(indices, base, n_digits)
        if scramble:
            digits = owen_scramble_digits(
                digits, base, np.random.default_rng(seed + dim)
            )
        points[:, dim] = digits_to_unit(digits, base)
    return points


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def cell_counts_1d(values: FloatArray, n_cells: int) -> IntArray:
    """Count how many ``values`` fall into each of ``n_cells`` equal bins of ``[0, 1)``."""
    index = np.clip((values * n_cells).astype(np.int64), 0, n_cells - 1)
    counts: IntArray = np.bincount(index, minlength=n_cells).astype(np.int64)
    return counts


def empty_cell_fraction(points: FloatArray, n_bins: int) -> float:
    """Return the fraction of empty cells on an ``n_bins x n_bins`` grid."""
    if points.shape[1] != 2:
        raise ValueError("points must be two-dimensional")
    index = np.clip((points * n_bins).astype(np.int64), 0, n_bins - 1)
    occupied: IntArray = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(occupied, (index[:, 0], index[:, 1]), 1)
    return float(np.mean(occupied == 0))


def leading_cell_occupancy(points: FloatArray, base_x: int, base_y: int) -> IntArray:
    """Count points per *leading-digit* cell, i.e. per ``base_x`` x ``base_y`` grid cell.

    The leading digit of ``phi_b(n)`` is what selects one of the ``b`` coarse
    bands, so this grid shows whether the first digits of the two coordinates
    have covered all ``base_x * base_y`` combinations.
    """
    if points.shape[1] != 2:
        raise ValueError("points must be two-dimensional")
    ix = np.clip((points[:, 0] * base_x).astype(np.int64), 0, base_x - 1)
    iy = np.clip((points[:, 1] * base_y).astype(np.int64), 0, base_y - 1)
    occupancy: IntArray = np.zeros((base_x, base_y), dtype=np.int64)
    np.add.at(occupancy, (ix, iy), 1)
    return occupancy


# ---------------------------------------------------------------------------
# Figure 1: one-dimensional stratification
# ---------------------------------------------------------------------------


def plot_stratification_base2(max_level: int = 4) -> Figure:
    """Show that the first ``2**m`` base-2 points fill the ``2**m`` dyadic cells once."""
    if max_level < 1:
        raise ValueError("max_level must be positive")

    fig = plt.figure(figsize=(12.5, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0), wspace=0.28)
    ax_cells = fig.add_subplot(grid[0, 0])
    ax_counts = fig.add_subplot(grid[0, 1])

    for level in range(1, max_level + 1):
        n_cells = 2**level
        y = float(level)
        points = van_der_corput(n_cells, 2)
        for k in range(n_cells):
            color = "#dbeafe" if k % 2 == 0 else "#eff6ff"
            ax_cells.add_patch(
                Rectangle(
                    (k / n_cells, y - 0.30),
                    1.0 / n_cells,
                    0.60,
                    facecolor=color,
                    edgecolor="#93c5fd",
                    linewidth=0.7,
                )
            )
        ax_cells.plot(
            points,
            np.full(points.shape, y),
            linestyle="none",
            marker="o",
            markersize=6.0,
            markerfacecolor="#dc2626",
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5,
        )
    ax_cells.set_xlim(-0.02, 1.02)
    ax_cells.set_ylim(0.4, float(max_level) + 0.6)
    ax_cells.set_yticks([float(level) for level in range(1, max_level + 1)])
    ax_cells.set_yticklabels([f"2^{level} points" for level in range(1, max_level + 1)])
    ax_cells.set_xlabel("x  (unit interval split into 2^m dyadic cells)")
    ax_cells.set_title(
        "Base-2 van der Corput: every cell receives exactly one point",
        fontsize=11,
    )
    ax_cells.text(
        0.02,
        0.03,
        "each dot sits at x = k / 2^m, the left edge of the cell it occupies",
        transform=ax_cells.transAxes,
        fontsize=8,
        color="#4b5563",
    )

    n_cells = 2**max_level
    counts = cell_counts_1d(van_der_corput(n_cells, 2), n_cells)
    cell_index = np.arange(n_cells)
    ax_counts.bar(cell_index, counts, width=0.82, color="#2563eb")
    ax_counts.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax_counts.set_ylim(0.0, 2.2)
    ax_counts.set_xlabel(f"dyadic cell index k  (m = {max_level}, {n_cells} cells)")
    ax_counts.set_ylabel("number of points in the cell")
    ax_counts.set_title(f"Counts at m = {max_level}: max = min = 1", fontsize=11)
    ax_counts.text(
        0.02,
        0.92,
        f"N = 2^m = {n_cells} points, {n_cells} cells, one point each",
        transform=ax_counts.transAxes,
        fontsize=9,
        color="#374151",
    )

    fig.suptitle(
        "1D Halton (base 2): perfect stratification at every resolution",
        fontsize=13,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: coprime versus non-coprime bases
# ---------------------------------------------------------------------------


def plot_coprime_effect(count: int = 512) -> Figure:
    """Compare 2D Halton bases that are coprime, share a factor, or are identical."""
    cases: list[tuple[int, int]] = [(2, 3), (2, 4), (2, 2)]
    fig = plt.figure(figsize=(13.0, 4.6))

    for position, (base_x, base_y) in enumerate(cases, start=1):
        ax: Axes = fig.add_subplot(1, 3, position)
        points = halton_points(count, (base_x, base_y))
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=6.0,
            color="#1d4ed8",
            alpha=0.85,
            linewidths=0.0,
        )
        gcd = int(np.gcd(base_x, base_y))
        empty = empty_cell_fraction(points, 16)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"$\\phi_{{{base_x}}}(n)$")
        ax.set_ylabel(f"$\\phi_{{{base_y}}}(n)$")
        ax.set_title(f"bases = ({base_x}, {base_y}),  gcd = {gcd}", fontsize=11)
        ax.text(
            0.03,
            0.96,
            f"empty 16x16 cells: {empty:.0%}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
        if gcd > 1:
            ax.set_facecolor("#fff7ed")

    fig.suptitle(
        "2D Halton: coprime bases fill the square, shared factors collapse onto a lattice",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Owen scrambling
# ---------------------------------------------------------------------------


def plot_scrambling_effect(count: int = 1024, seed: int = DEFAULT_SEED) -> Figure:
    """Compare plain and Owen-scrambled 2D Halton for a low and a high base pair."""
    cases: list[tuple[int, int, str]] = [
        (2, 3, "dimensions 1-2 (bases 2, 3)"),
        (11, 13, "dimensions 5-6 (bases 11, 13)"),
    ]
    fig = plt.figure(figsize=(11.0, 10.4))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.20)

    for row, (base_x, base_y, label) in enumerate(cases):
        plain = halton_points(count, (base_x, base_y))
        scrambled = halton_points(count, (base_x, base_y), scramble=True, seed=seed)
        for col, (points, kind, color) in enumerate(
            ((plain, "plain", "#1d4ed8"), (scrambled, "scrambled", "#b45309"))
        ):
            ax: Axes = fig.add_subplot(grid[row, col])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=4.0,
                color=color,
                alpha=0.85,
                linewidths=0.0,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.set_xlabel(f"$\\phi_{{{base_x}}}(n)$")
            ax.set_ylabel(f"$\\phi_{{{base_y}}}(n)$")
            ax.set_title(f"({base_x}, {base_y})  {kind}", fontsize=11)
            ax.text(
                0.03,
                0.97,
                f"{label}\nempty 16x16 cells: {empty_cell_fraction(points, 16):.0%}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#374151",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
            )
            if row == 0 and kind == "plain":
                ax.scatter(
                    [0.0],
                    [0.0],
                    s=140.0,
                    facecolors="none",
                    edgecolors="#dc2626",
                    linewidths=1.6,
                    clip_on=False,
                )
                ax.text(
                    0.06,
                    0.07,
                    "n = 0 is pinned to the corner",
                    fontsize=8,
                    color="#dc2626",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                        "pad": 1.0,
                    },
                )

    fig.suptitle(
        f"Owen scrambling (seed = {seed}): same coverage, randomised assignment, no stripes",
        fontsize=13,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: where the stripes come from
# ---------------------------------------------------------------------------


def _style_stripe_axes(ax: Axes, base_x: int, base_y: int) -> None:
    """Apply the square, labelled style shared by the stripe panels."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(f"$\\phi_{{{base_x}}}(n)$", fontsize=8)
    ax.set_ylabel(f"$\\phi_{{{base_y}}}(n)$", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_stripe_mechanism(count: int = 1024) -> Figure:
    """Show why unscrambled Halton lines up on diagonals for large bases."""
    fig = plt.figure(figsize=(14.0, 9.0))
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.20)

    # (a) mechanism: the first 11 points satisfy y = (11 / 13) x exactly
    ax_line: Axes = fig.add_subplot(grid[0, 0])
    first = halton_points(121, (11, 13))
    line_x = np.linspace(0.0, 1.0, 200)
    slope = 11.0 / 13.0
    for offset in np.arange(-0.3, 1.2, 1.0 / 13.0):
        ax_line.plot(
            line_x,
            slope * line_x + offset,
            color="#dc2626",
            linewidth=0.5,
            alpha=0.5,
        )
    ax_line.scatter(first[11:, 0], first[11:, 1], s=14.0, color="#1d4ed8", linewidths=0.0)
    ax_line.plot(
        first[:11, 0],
        first[:11, 1],
        "-o",
        color="#f59e0b",
        markersize=3.5,
        linewidth=1.4,
        zorder=4,
    )
    ax_line.annotate(
        "first 11 points:\ny = (11/13) x",
        xy=(0.45, 0.38),
        xytext=(0.52, 0.12),
        fontsize=8,
        color="#b45309",
        arrowprops={"arrowstyle": "->", "color": "#b45309", "linewidth": 0.9},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
    )
    ax_line.set_title("(11, 13), N = 121: points follow slope 11/13", fontsize=9)
    _style_stripe_axes(ax_line, 11, 13)

    # (b) N = 11 * 13: the leading-digit grid is exactly full
    ax_full: Axes = fig.add_subplot(grid[0, 1])
    full = halton_points(11 * 13, (11, 13))
    occupancy = leading_cell_occupancy(full, 11, 13)
    for k in range(1, 11):
        ax_full.axvline(k / 11.0, color="#cbd5e1", linewidth=0.4, alpha=0.7, zorder=0)
    for k in range(1, 13):
        ax_full.axhline(k / 13.0, color="#cbd5e1", linewidth=0.4, alpha=0.7, zorder=0)
    ax_full.scatter(full[:, 0], full[:, 1], s=14.0, color="#1d4ed8", linewidths=0.0)
    ax_full.set_title("(11, 13), N = 143 = 11x13", fontsize=9)
    ax_full.text(
        0.03,
        0.96,
        f"{int(np.count_nonzero(occupancy))}/143 leading cells filled, once each",
        transform=ax_full.transAxes,
        va="top",
        fontsize=8,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
    )
    _style_stripe_axes(ax_full, 11, 13)

    # (c) N = 1024: the next digit re-creates the lines at a finer scale
    ax_more: Axes = fig.add_subplot(grid[0, 2])
    more = halton_points(count, (11, 13))
    ax_more.scatter(more[:, 0], more[:, 1], s=2.0, color="#1d4ed8", linewidths=0.0)
    ax_more.set_title(f"(11, 13), N = {count}: stripes at the next level", fontsize=9)
    _style_stripe_axes(ax_more, 11, 13)

    # bottom row: same N, increasing base -> sharper lines
    for column, (base_x, base_y) in enumerate(((2, 3), (19, 23), (97, 101))):
        ax: Axes = fig.add_subplot(grid[1, column])
        points = halton_points(count, (base_x, base_y))
        ax.scatter(points[:, 0], points[:, 1], s=2.0, color="#1d4ed8", linewidths=0.0)
        empty = empty_cell_fraction(points, 16)
        ax.set_title(f"({base_x}, {base_y}), N = {count}", fontsize=9)
        ax.text(
            0.03,
            0.96,
            f"empty 16x16 cells: {empty:.0%}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
        )
        _style_stripe_axes(ax, base_x, base_y)

    fig.suptitle(
        "Stripes: leading digits of both coordinates advance in lockstep; "
        "larger bases sharpen the lines",
        fontsize=12,
    )
    return fig


def stripe_polylines(
    points: FloatArray, base_x: int, base_y: int, min_run: int = 3
) -> list[FloatArray]:
    """Group consecutive points that lie on the same stripe into polylines.

    A stripe is a maximal run of consecutive indices during which both leading
    digits advance without wrapping (``n % base_x != 0`` and
    ``n % base_y != 0``).  Runs with fewer than ``min_run`` points are dropped,
    so a base pair whose digits wrap every other step (e.g. ``(2, 3)``)
    produces no stripe at all.

    Returns:
        One ``(n_points_in_run, 2)`` array per stripe.
    """
    n_points = points.shape[0]
    if n_points < 2:
        return []

    steps = np.arange(1, n_points)
    breaks = np.where((steps % base_x == 0) | (steps % base_y == 0))[0]
    starts = np.concatenate((np.array([0]), breaks + 1))
    ends = np.concatenate((breaks + 1, np.array([n_points])))

    stripes: list[FloatArray] = []
    for start, end in zip(starts, ends):
        if end - start >= min_run:
            stripes.append(points[start:end])
    return stripes


def plot_stripe_base_comparison(count: int = 1024) -> Figure:
    """Draw the same number of points for three base pairs side by side.

    The orange segments join consecutive points that share a stripe.  ``(2, 3)``
    produces no segments (the digits wrap every other step), ``(11, 13)`` draws
    clear diagonals and ``(19, 23)`` sharpens them further: with a larger base
    the first digit outweighs the rest by ``1 / b``, so the lockstep line
    survives.
    """
    cases: list[tuple[int, int]] = [(2, 3), (11, 13), (19, 23)]
    fig = plt.figure(figsize=(14.0, 4.9))

    for column, (base_x, base_y) in enumerate(cases, start=1):
        ax: Axes = fig.add_subplot(1, 3, column)
        points = halton_points(count, (base_x, base_y))
        stripes = stripe_polylines(points, base_x, base_y)
        if stripes:
            ax.add_collection(
                LineCollection(stripes, colors="#f59e0b", linewidths=0.45, alpha=0.6)
            )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=2.0,
            color="#1d4ed8",
            linewidths=0.0,
            zorder=3,
        )
        ax.set_title(f"bases ({base_x}, {base_y}), N = {count}", fontsize=10)
        ax.text(
            0.03,
            0.96,
            f"first-digit weight  1/{base_x} = {1.0 / base_x:.2f}\n"
            f"stripes drawn: {len(stripes)}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
        )
        _style_stripe_axes(ax, base_x, base_y)

    fig.suptitle(
        f"Same budget N = {count}: larger bases line the points up on diagonals",
        fontsize=12,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the five figures, then show them or save them to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="write PNG files to this directory instead of calling plt.show()",
    )
    args = parser.parse_args()
    save_dir: Path | None = args.save_dir

    figures: list[tuple[str, Figure]] = [
        ("halton_1d_stratification", plot_stratification_base2()),
        ("halton_2d_coprime", plot_coprime_effect()),
        ("halton_2d_scrambling", plot_scrambling_effect()),
        ("halton_2d_stripes", plot_stripe_mechanism()),
        ("halton_2d_stripe_bases", plot_stripe_base_comparison()),
    ]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = save_dir / f"{name}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            print(f"wrote {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
