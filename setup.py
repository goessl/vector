from setuptools import setup
  
setup(
    name = 'hermite_function',
    version = '0.9',
    description = 'A Hermite function series module.',

    author = 'Sebastian Gössl',
    author_email = 'goessl@student.tugraz.at',
    license = 'MIT',

    url = 'https://github.com/goessl/hermite_function',
    py_modules = ['HermiteFunction'],
    python_requires = '>=3.7',
    install_requires = ['numpy', 'scipy'],

    classifiers = [
      'Programming Language :: Python :: 3.7',
      'License :: OSI Approved :: MIT License'
    ]
)
