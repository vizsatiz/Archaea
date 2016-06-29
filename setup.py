import setuptools

__version__ = '0.0.1'

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
        'Development Status :: Beta-1',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7'
    ],
)