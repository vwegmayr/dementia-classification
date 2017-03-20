from setuptools import setup
from settings import AUTHOR, EMAIL, PROJECT


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name=PROJECT,
      version='0.0.1',
      description='Description',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: topic1 :: topic2 :: topic3',
      ],
      keywords='kw1 kw2 kw3',
      url='http://github.com/storborg/funniest',
      author=AUTHOR,
      author_email=EMAIL,
      license='MIT',
      packages=[PROJECT,
      ],
      install_requires=[
          'PyYAML',
      ],
      #tests_require=[] in case test have specific requirements
      entry_points={
          'console_scripts': ['yourcommand='+PROJECT+'.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
