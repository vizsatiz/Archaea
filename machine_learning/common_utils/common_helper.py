import numpy


def add_column_of_ones(matrix, M):
    new_col = numpy.ones((M, 1))
    matrix_with_bias = numpy.hstack((new_col, matrix))
    return matrix_with_bias
