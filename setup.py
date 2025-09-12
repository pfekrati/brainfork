from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brainfork",
    version="0.1.2",
    author="Pooyan",
    author_email="",  # Add your email here
    description="An intelligent AI model router for Azure OpenAI that automatically selects the best model for your specific use case",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pfekrati/brainfork",  # Update with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="azure openai ai model router llm gpt",
    project_urls={
        "Bug Reports": "https://github.com/pfekrati/brainfork/issues",
        "Source": "https://github.com/pfekrati/brainfork",
        "Documentation": "https://github.com/pfekrati/brainfork#readme",
    },
)