from setuptools import setup

setup(
    name="AIStorch_ex",
    version="0.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/torch_AIS_ex_module.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["torch_AIS_ex"],
    zip_safe=False,
    install_requires=[
        "torch", "python-ex"
    ]
)
