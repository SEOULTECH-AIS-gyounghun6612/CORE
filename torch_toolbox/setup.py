from setuptools import setup, find_namespace_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, encoding="utf-8") as f:
        return [_l for _l in f.readlines() if _l and not _l[0] != "#"]


setup(
    name="torch_ex",
    version="0.0.1",
    description="Custom base code module for pytorch",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    python_requires=">=3.10.0",
    install_requires=load_requirements(),
    zip_safe=False,
)
