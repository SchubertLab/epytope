from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from distutils.core import Extension
from codecs import open  # To use a consistent encoding
from os import path
import glob

here = path.abspath(path.dirname(__file__))

#for packaging files must be in a package (with init) and listed in package_data
# package-externals can be included with data_files,
# and there is a bug in patter nmatching http://bugs.python.org/issue19286
# install unclear for data_files

# Read the contents of the README.md file for use as long_description
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='epytope',

    # Version:
    version='3.1.0',

    description='A Framework for Epitope Detection and Vaccine Design',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/KohlbacherLab/epytope',

    # Author details
    author='Benjamin Schubert, Mathias Walzer, Christopher Mohr, Leon Kuchenbecker',
    author_email='benjamin.schubert@helmholtz-muenchen.de, walzer@ebi.ac.uk, christopher.mohr@uni-tuebingen.de, leon.kuchenbecker@uni-tuebingen.de ',

    # maintainer details
    maintainer='Christopher Mohr',
    maintainer_email='christopher.mohr@uni-tuebingen.de',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # The license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    # What epytope relates to:
    keywords='epitope prediction vaccine design HLA MHC',

    # Specify  packages via find_packages() and exclude the tests and 
    # documentation:
    packages=find_packages(),

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #include_package_data=True,
    package_data={
            'epytope.Data.examples': ['*.*'],
            'epytope.Data.svms.svmtap': ['*'],
            'epytope.Data.svms.svmhc': ['*'],
            'epytope.Data.svms.unitope': ['*'],
    },

    data_files = [
            ('docs', ['CHANGELOG.md']),
            ],

    #package_data is a lie: http://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute

    # 'package_data' is used to also install non package data files
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # example:
    #data_files=data_files,

    # Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # IMPORTANT: script names need to be in lower case ! ! ! (otherwise 
    # deinstallation does not work)
    #entry_points={
    #    'console_scripts': [
    #        'epitopeprediction=epytope.Apps.EpitopePrediction:main',
    #    ],
    #},

    # Run-time dependencies. (will be installed by pip when epytope is installed)
    # TODO: find alternative for SMVlight scikitlearn
    install_requires=[
            'setuptools<=57',
            'pandas',
            'pyomo>=4.0',
            'PyMySQL',
            'biopython',
            'pyVCF',
            'mhcflurry<=1.4.3',
            'mhcnuggets',
            'keras<=2.3.1',         # legacy tensorflow required by mhcnuggets resolves to incompatible keras version
            'h5py<=2.10.0',         # mhcnuggets fails to read model with newer versions
            ],

)
