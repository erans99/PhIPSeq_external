from distutils.core import setup

setup(
    name='PhIPSeq_external',
    version='0.1.0',
    author='Sigal',
    author_email='speledleviatan@gmail.com',
    packages=['PhIPSeq_external'],
    license='LICENSE',
    long_description=open('README.md').read(),
    requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'statsmodels', 'seaborn',
              'statannot'])
