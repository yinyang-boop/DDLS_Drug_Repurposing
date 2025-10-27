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
        'rdkit',  # 核心依赖
        'tqdm',   # 如果您的数据处理包含进度条
        # 完整的依赖列表应放在 requirements.txt 中
    ],
    author='Your Name',
    description='用于药物再利用项目的分子计算流水线工具箱。',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/yourrepo' # 你的GitHub链接
)