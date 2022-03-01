from setuptools import setup

setup(
    name="python_ex",
    version="1.0.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["python_ex"],
    zip_safe=False,
    python_requires=">= 3.7.0",
    install_requires=[
        'numpy', 'opencv-python'
    ]
)
