import setuptools

#Use README.md for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Official implementation of the ProVADA package for conditional protein variant design."

# Read dependencies from requirements.txt
try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found. Installing without dependencies.")
    requirements = []


setuptools.setup(
    # --- Project Metadata ---
    name="provada",
    version="1.0.0",
    author="Sophia Lu, Ben Viggiano, Xiaowei Zhang",
    author_email="sophialu@stanford.edu, viggiano@stanford.edu, zhangxw@stanford.edu",
    description="Official implementation of the ProVADA package for conditional protein variant design.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    
    # --- Project URLs ---
    url="https://github.com/SUwonglab/provada",
    project_urls={
        "Homepage": "https://github.com/SUwonglab/provada",
        "Bug Tracker": "https://github.com/SUwonglab/provada/issues",
        "Publication": "https://www.biorxiv.org/content/10.1101/2025.07.11.664238v1",
    },

    # --- Build Configuration ---
    packages=["provada"],
    
    # --- Dependencies ---
    python_requires=">=3.11",
    install_requires=requirements,

    # --- Classifiers for PyPI ---
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
