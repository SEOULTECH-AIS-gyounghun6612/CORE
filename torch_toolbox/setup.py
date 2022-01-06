from setuptools import setup

setup(
    name="torch_ex",
    version="0.0.1",
    description="Custom base code module for pytorch",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/torch_AIS_ex_utils.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["torch_ex"],
    zip_safe=False,
    install_requires=[
        "torch", "python_ex"
    ]
)
