from setuptools import find_packages, setup

with open("README.md") as f:
    long_description= f.read()

setup(
    name="StrucPy",
    version="0.0.1",
    description="Object Oriented Structural Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TabishIzhar/StrucPy.git",
    author="Tabish Izhar",
    author_email="tizhar@iul.ac.in",
    license="GNU LESSER GENERAL PUBLIC LICENSE v2.1 or later (GNU LGPLv2.1)",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    package_dir= {"":"src"},
    packages=find_packages(where='src'),
    include_package_data=False,
    install_requires=["numpy==1.23.3", "pandas==1.4.4", "plotly==5.10.0", "ray==2.6.1", "openpyxl==3.0.10" ],
    extras_require={
        "dev": ["pytest>=7.0","twine>=4.0.2"],
    },
    python_requires=">=3.10",
)



