from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='subtyping',
      version='0.0.1',
      description='Tools for subtype identification in structured and image data',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Clustering :: Structured Data :: Medical Image Analysis',
      ],
      keywords='embedding autoencoder clustering',
      url='http://github.com/storborg/funniest',
      author='AUTHOR',
      author_email='author@gmail.com',
      license='MIT',
      packages=['subtyping',
      ],
      install_requires=[
          'PyYAML',
      ],
      #tests_require=[] in case test have specific requirements
      entry_points={
          'console_scripts': ['sti=subtyping.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)