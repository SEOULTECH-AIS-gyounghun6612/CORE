from setuptools import setup

setup(
    name="python_AIS_ex_utils",
    version="2.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/python_AIS_ex_utils.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["python_AIS_ex_utils"],
    zip_safe=False,
    install_requires=[
        "opencv-python", "numpy"
    ]
)
