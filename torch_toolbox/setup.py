from setuptools import setup

setup(
    name="AIS_torch_utils",
    version="1.0.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/python_base_utils.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["utils"],
    zip_safe=False,
    install_requires=[
        "torch", "numpy"
    ]
)
