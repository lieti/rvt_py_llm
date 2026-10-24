"""
0.1.1 fix X5 (docs/KNOWN_ISSUES.md KI-36): BlenderLayer.check_data rejects an opacity outside [0, 100].

At v0.1.0 the check read ``if 0 > self.opacity > 100``, a chained comparison that needs opacity < 0 and > 100 at once,
so every opacity passed and values above 100 extrapolated in rendering (KI-22). The reading of FIX_BRIEF_LLM.md
section 1: ``if not 0 <= opacity <= 100: raise``, with the exception type of the neighbouring checks (``Exception``).

Expected values: the interval [0, 100] written in the test, and the exception type raised by the other checks of
check_data (asserted in test_neighbouring_checks_raise_exception, so a change there is seen here). Synthetic data only.
"""

import numpy as np
import pytest

import rvt.blend as rb


def _layer(opacity, **kwargs):
    args = dict(vis_method="Hillshade", normalization="value", minimum=0, maximum=1, blend_mode="normal",
                opacity=opacity)
    args.update(kwargs)
    return rb.BlenderLayer(**args)


def _raised_type(func):
    try:
        func()
    except Exception as exc:  # noqa: BLE001
        return type(exc)
    return None


def test_neighbouring_checks_raise_exception():
    assert _raised_type(_layer(50, blend_mode="dissolve").check_data) is Exception
    assert _raised_type(_layer(50, min_colormap_cut=0.2).check_data) is Exception


@pytest.mark.parametrize("opacity", [150, 100.5, 101, -5, -0.001, 1e9, -np.inf, np.inf])
def test_opacity_outside_0_100_raises(opacity):
    with pytest.raises(Exception, match="opacity") as info:
        _layer(opacity).check_data()
    assert info.type is _raised_type(_layer(50, blend_mode="dissolve").check_data)


@pytest.mark.parametrize("opacity", [0, 0.0, 0.3, 1, 50, 99.9, 100, 100.0, np.float32(100), np.int64(0)])
def test_opacity_inside_0_100_is_accepted(opacity):
    _layer(opacity).check_data()


def test_empty_layer_is_not_checked():
    layer = rb.BlenderLayer(None, opacity=500)  # vis None: every attribute is reset, nothing raises
    layer.check_data()
    assert layer.opacity is None


def test_combination_check_data_and_render_all_images_raise():
    # KI-36 repro: a layer at 150 % over a layer at 100 %
    comb = rb.BlenderCombination()
    comb.create_layer("x", "value", 0, 1, "normal", 150, image=np.full((1, 2), 0.6))
    comb.create_layer("y", "value", 0, 1, "normal", 100, image=np.full((1, 2), 0.4))
    with pytest.raises(Exception, match="opacity") as info:
        comb.check_data()
    assert info.type is Exception
    with pytest.raises(Exception, match="opacity") as info:
        comb.render_all_images()
    assert info.type is Exception
