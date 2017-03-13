from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='globaltracking',
      version='0.0.1',
      description='Global tractography on multimodal diffusion data.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Diffusion MRI Processing :: Tractography',
      ],
      keywords='tractography mri diffusion',
      url='http://github.com/storborg/funniest',
      author='Luca Wolf',
      author_email='luca.wolf@bluemail.ch',
      license='MIT',
      packages=['globaltracking',
                'globaltracking.energy'],
      install_requires=[
          'PyYAML',
      ],
      #tests_require=[] in case test have specific requirements
      entry_points={
          'console_scripts': ['fiber-tracking=globaltracking.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)