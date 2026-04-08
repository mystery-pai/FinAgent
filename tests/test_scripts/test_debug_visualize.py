def test_build_visualization_filters_with_year():
    from app.ui.workflows import build_visualization_filters

    assert build_visualization_filters(2024) == {"year": 2024}


def test_build_visualization_filters_without_year():
    from app.ui.workflows import build_visualization_filters

    assert build_visualization_filters(None) is None
