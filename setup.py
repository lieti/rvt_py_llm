import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rvt_py_llm",
    version="0.2.0",
    author="ZRC SAZU and University of Ljubljana (UL FGG)",
    author_email="ziga.kokalj@zrc-sazu.si",
    maintainer="Tibor Lieskovský",
    maintainer_email="tibor.lieskovsky@stuba.sk",
    description="Drop-in of rvt_py, the Relief Visualization Toolbox Python library (import name rvt): "
                "same API and bitwise identical outputs, faster, documented and tested.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EarthObservation/RVT_py",
    packages=["rvt"],
    # SPDX expression; replaces the license classifier, which setuptools >= 77 reports as deprecated.
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    keywords="relief_visualization_toolbox relief_visualization relief rvt raster raster_visualization visualization",
    # The code uses no syntax newer than Python 3.9 (checked with ast.parse(feature_version=(3, 9))) and
    # numba >= 0.60 needs Python >= 3.9. Tested on Python 3.11 only.
    python_requires=">=3.9",
    project_urls={
        "Upstream source": "https://github.com/EarthObservation/RVT_py",
        "Upstream documentation": "https://rvt-py.readthedocs.io/en/latest/",
        "Old RVT": "https://iaps.zrc-sazu.si/en/rvt#v",
        "ArcGIS Pro": "https://github.com/EarthObservation/rvt-arcgis-pro",
        "QGIS plugin": "https://github.com/EarthObservation/rvt-qgis",
    },
    # The runtime imports of rvt/: numpy, scipy (rvt.vis), matplotlib (rvt.blend_func), osgeo.gdal
    # (rvt.default, rvt.tile). numba is optional (rvt._accel imports it lazily and falls back to numpy).
    install_requires=["numpy", "scipy", "matplotlib", "gdal"],
    extras_require={
        "accel": ["numba>=0.60"],
        # rasterio is used by tests/test_char_default.py and tests/test_char_tile.py, not by rvt/.
        "test": ["pytest", "rasterio"],
    },
)
