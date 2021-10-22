from setuptools import setup

setup(
    name="python_ex",
    version="2.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/python_ex_module.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["python_ex"],
    zip_safe=False,
    install_requires=[
        "opencv-python", "numpy"
    ]
)
