from setuptools import find_packages, setup

setup(
    name="gaze_rs",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "hydra-core",
        "wandb",
        "numpy",
        "opencv-python",
        "imageio",
        "seaborn",
        "ruff",
    ],
)
