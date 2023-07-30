from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()
  
setup(
    name = 'hermite-function',
    version = '1.1',
    description = 'A Hermite function series module.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    
    author = 'Sebastian Gössl',
    author_email = 'goessl@student.tugraz.at',
    license = 'MIT',
    
    py_modules = ['hermitefunction'],
    url = 'https://github.com/goessl/hermite-function',
    python_requires = '>=3.7',
    install_requires = ['numpy', 'scipy'],
    
    classifiers = [
      'Programming Language :: Python :: 3.7',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent'
    ]
)
