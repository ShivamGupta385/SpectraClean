from setuptools import setup, find_packages

setup(
    name="spectraclean",
    version="1.0.0",
    description="Hyperspectral Image (HSI) Mixed Noise Denoising Pipeline using Fast RPCA and PCA-NLM Filtering",
    author="Shivam Gupta",
    url="https://github.com/ShivamGupta385/SpectraClean",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
