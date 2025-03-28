from setuptools import setup, find_packages

setup(
    name="smart_traffic_management",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "opencv-python>=4.7.0.72",
        "tensorflow>=2.11.0",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.3",
        "pillow>=9.3.0",
        "matplotlib>=3.5.3",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Smart AI-based Traffic Management System",
    keywords="traffic, AI, management, smart city",
    python_requires=">=3.8",
)
