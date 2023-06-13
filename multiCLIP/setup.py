"""Configure the package for installation."""
from setuptools import find_packages, setup

setup(
    name="video_retrieval",
    version="0.0.1",
    description=("Toolkit for MultiVENT video research."),
    url="https://github.com/katesanders9/multiVENT/tree/main/multiCLIP",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "albumentations>=0.4.5",
        "tensorboard>=2.2.2",
        "tensorboardX>=2.0",
        "torch==1.12.0",
        "torchvision==0.13.0",
        "tqdm>=4.43.0",
        "numpy>=1.20.0",
        "einops>=0.3.0",
        "imgaug>=0.4.0",
        "transformers>=4.11.0",
        "tokenizers>=0.10.3",
        "datasets>=1.14.0",
        "decord>=0.6.0",
        "timm>=0.4.12",
        "av>=8.0.3",
        "psutil>=5.8.0",
        "wget>=3.2",
        "yt_dlp",
        "pandas",
        "hydra-core",
        "omegaconf",
        "open-clip-torch>=2.16.0",
        "hydra-core>=1.3.2",
        "urllib3==1.26.6",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "packaging",
            "pylint",
            "pytest",
            "pytest-mock",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "video-retrieval-openclip-video-infer=video_retrieval.cli.openclip_video_infer:main",
        ]
    },
)
