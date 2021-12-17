from setuptools import setup

setup(
    name="python_ex",
    version="0.0.1",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/python_AIS_ex_utils.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["python_ex"],
    zip_safe=False,
    install_requires=[
        'python >= "3.7"', 'numpy', 'opencv-python'
    ]
)
