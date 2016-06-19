import setuptools

__version__ = 'alpha-0.1'

setuptools.setup(

    name="archaea",
    version=__version__,
    url="https://github.com/archaea-solutions/archaea.git",

    author="vishnu satis",
    author_email="satis.vishnu@gmail.com",

    description="AI as a Service",
    long_description=open('README.rst').read(),

    py_modules=['machine_learning', 'machine_learning_tests'],
    zip_safe=False,
    platforms='any',

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
)