from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tcgai-toolkit",
    version="0.3.0",
    author="TCG AI PRO",
    description="Python toolkit for TCG card image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CeferinSoftware/tcgai-toolkit",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Development Status :: 4 - Beta",
    ],
)