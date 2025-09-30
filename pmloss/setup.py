from setuptools import setup, find_packages

# Read the content of requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read the content of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pmloss",  # The package name
    version="0.1.0",
    author="Duochao Shi", 
    author_email="dcshi@zju.edu.cn", 
    description="PM-Loss, a novel regularization loss based on a learned point map for feed-forward 3DGS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aim-uofa/PM-Loss",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=required,
)