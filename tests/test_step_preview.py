import importlib.util
import os

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cadquery") is None,
    reason="cadquery is not installed.",
)


def test_render_step_preview_creates_png(tmp_path):
    from PIL import Image
    from scripts.render_step_preview import render_step_preview

    output = tmp_path / "preview.png"
    result = render_step_preview(
        "tests/fixtures/simple_block.stp",
        str(output),
        width=320,
        height=240,
    )

    assert result == str(output.resolve())
    assert os.path.exists(result)
    image = Image.open(result)
    assert image.size == (320, 240)
