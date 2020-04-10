#!/usr/bin/env python

# The library version number is repeated here.
# see e.g. https://packaging.python.org/en/latest/single_source_version/#single-sourcing-the-version

from setuptools import setup


setup(name='twitter_data_collection',
      version='0.3.2',
      description='twitter_data_collection',
      author='Sanjeev Kumar Karn and Mark Buckley',
      author_email='skarn@cis.lmu.de',
      package_dir={'': 'src/main/python'},
      packages=['twitter_data_collection', 'twitter_data_collection.etc'],
      install_requires=[
            'tweepy',
            'newspaper'
      ],
      package_data={'twitter_data_collection.etc': ['twitter_auth.conf', 'logging.conf']},
      entry_points={
       'console_scripts': [
             'twitter_read_stream = twitter_data_collection.read_twitter_stream:main',
             'transform_data_format = twitter_data_collection.transform_data_format:main'
       ]
      }

      )
