from setuptools import setup, find_namespace_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, encoding="utf-8") as f:
        return [_l for _l in f.readlines() if _l and not _l[0] != "#"]


setup(
    name="python_ex",
    version="1.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    python_requires=">= 3.12.0",
    install_requires=load_requirements(),
    zip_safe=False
)
