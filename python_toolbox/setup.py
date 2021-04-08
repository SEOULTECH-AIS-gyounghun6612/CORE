from setuptools import setup

setup(
    name="AIS_utils",
    version="0.0.1",
    description="Custom code module in python",
    url="https://github.com/gyounghun6612/base_utils_v2.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["ais_utils"],
    zip_safe=False,
    install_requires=[
        "opencv-python"
    ]
)

setup(
    name="AIS_utils",
    version="0.0.1",
    description="Custom code module in python",
    url="https://github.com/gyounghun6612/base_utils_v2.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["ais_gui"],
    zip_safe=False,
    install_requires=[
        "opencv-python",
        "ais_utils"
    ]
)
