from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autism-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Autism Detection & Emotion Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autism-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "autism-detection=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.h5", "*.pkl", "*.joblib"],
    },
    data_files=[
        ("models", ["models/autism_detector.h5", "models/emotion_analyzer.h5"]),
    ],
) 