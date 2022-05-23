#!/usr/bin/env python
# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt"), "r") as req:
    required_packages = [ln.strip() for ln in req.readlines()]

# Load packages from requirements-dev.txt
with open(Path(BASE_DIR, "requirements-dev.txt"), "r") as req_dev:
    dev_packages = [ln.strip() for ln in req_dev.readlines()]

# Load packages from requirements-doc.txt
with open(Path(BASE_DIR, "requirements-doc.txt"), "r") as req_doc:
    doc_packages = [ln.strip() for ln in req_doc.readlines()]


setup(
    name="eda-cv",
    version="1.0",
    license="",
    description="installation du projet",
    author="MKL",
    author_email="klimczak.mathieu@pm.me",
    url="",
    # package_dir={"": "src"},
    # packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[required_packages],
    extras_require={
        "dev": [dev_packages] + [doc_packages],
        "docs": [doc_packages],
    },
)
