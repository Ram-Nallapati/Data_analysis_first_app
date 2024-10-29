from setuptools import setup, find_packages

setup(
    name="Data_analysis_first_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.26.0",
        "pandas==1.5.3",
        "numpy==1.23.5",
        # other dependencies
    ],
    include_package_data=True,
    zip_safe=False,
)
