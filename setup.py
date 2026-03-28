from setuptools import setup, find_packages

setup(
    name="geochemad",
    version="0.1.0",
    description="GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration",
    author="Based on Ding et al. (2026)",
    url="https://github.com/RichardScottOZ/GeochemAD",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "geochemad-benchmark=scripts.run_benchmark:main",
        ],
    },
)
