from setuptools import setup, find_namespace_packages

setup(
    name="python_ex",
    version="1.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    python_requires=">= 3.12.0",
    zip_safe=False
)
