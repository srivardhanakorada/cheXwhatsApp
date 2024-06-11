import setuptools
setuptools.setup(
    name='cheXwhatsApp',
    version='0.1.2',
    author="Sahil Lathiya",
    author_email="sahillathiya456@gmail.com",
    description="",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchvision', 'opencv-python', 'pillow', 'imageio', 'scikit-image', 'numpy', 'tqdm', 'pandas'],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    include_package_data=True,
    data_files=[('models', ['cheXwhatsApp/lung_segmentation.pt'])],  # Explicitly include .pt file (modify path if needed)

)