"""SciencePlots journal look without ``text.usetex`` (no external LaTeX required).

The upstream ``science`` matplotlib style sets ``text.usetex`` to True. Mixed text/math
labels then depend on a working LaTeX toolchain; without it, ROC AUC and similar strings
often fail or look wrong. Disabling usetex keeps the same palette/typography intent while
using matplotlib's built-in mathtext for ``$...$`` segments.
"""
from __future__ import annotations

from contextlib import contextmanager


def try_apply_science_style() -> bool:
    """
    Apply globally after ``matplotlib.use(...)`` and ``import matplotlib.pyplot as plt``.

    Returns True if the science style is active.
    """
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return False
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("science")
    mpl.rcParams["text.usetex"] = False
    return True


@contextmanager
def science_style_context():
    """Use for a single figure; re-applies science and clears usetex inside the block."""
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        yield
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with plt.style.context("science"):
        mpl.rcParams["text.usetex"] = False
        yield
