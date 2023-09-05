from setuptools import setup, find_packages

version = "0.0.4"
description = "A vector database for face embeddings or encodings"

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

name = "FaceDB"
author = "sifat (shhossain)"

with open("requirements.txt") as f:
    required = f.read().splitlines()

keywords = ["python", "face", "recognition"]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Linguistic",
    "Topic :: Utilities",
    "Operating System :: OS Independent",
]

projects_links = {
    "Documentation": "https://github.com/shhossain/facedb",
    "Source": "https://github.com/shhossain/facedb",
    "Bug Tracker": "https://github.com/shhossain/facedb/issues",
}


setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    url="https://github.com/shhossain/facedb",
    project_urls=projects_links,
    packages=find_packages(),
    install_requires=required,
    keywords=keywords,
    classifiers=classifiers,
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
)
