from setuptools import setup, find_packages

setup(
    name='mcp-pipeline',
    version='0.1.0',
    packages=find_packages(),
    
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'rdkit',  
        'plotly', 
        'torch', 
        'prefect>=2.0', 
        'scikit-learn',
        'requests' 
    ],
    
    author='Yin Yang',
    description='A modular computational pipeline (MCP) toolset for Drug-Target Affinity prediction and cheminformatics analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yinyang-boop/DDLS_Drug_Repurposing',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
