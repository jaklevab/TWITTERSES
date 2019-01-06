from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension("fast_pt_in_poly", ["fast_pt_in_poly.pyx"],
              include_dirs=["/home/jlevyabi/seacabo/geoanaconda/anaconda3/include"],
              library_dirs=["/home/jlevyabi/seacabo/geoanaconda/anaconda3/lib"],
              libraries=["geos_c"])  # à renommer selon les besoins
]

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = cythonize(extensions),
)

