"""
 MikuAi - Is a toolkit for creating Conversational AI applications.
 Copyright by Sebastian Bürger [sebidev]
"""

import setuptools

setuptools.setup(
    name = 'mikuai',
    packages = ['mikuai'],
    version = '1.0.0',
    license='BSD 3-Clause License',
    description = 'Is a toolkit for creating Conversational AI applications.',
    author = 'sebidev',
    author_email = 'sebidev@hotmail.com',
    url="http://shizukaai.github.io",
    packages=setuptools.find_packages(),
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='mikuai machine learning ai ki deepspeech voice synthesis',
    python_requires='>=3.7',
)
