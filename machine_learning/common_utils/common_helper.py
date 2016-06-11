import numpy
import error_messages as error


def add_column_of_ones(matrix, m):
    """
    This method adds a column of ones in front of any given matrix

    :param matrix:
    :param m:
    :return:
    """
    new_col = numpy.ones((m, 1))
    matrix_with_bias = numpy.hstack((new_col, matrix))
    return matrix_with_bias


def validated_x_y_theta_dimensions(x, y, theta):
    """
    Validates the condition between x, y and theta so as it is goo enough to perform
    all operations of regression

    :param x:
    :param y:
    :param theta:
    :return:
    """
    shape_of_x = numpy.shape(x)
    number_of_rows_of_x = shape_of_x[0]
    number_of_columns_of_x = shape_of_x[1]
    shape_of_y = numpy.shape(y)
    number_of_rows_of_y = shape_of_y[0]
    number_of_columns_of_y = shape_of_y[1]
    shape_of_theta = numpy.shape(theta)
    number_of_rows_of_theta = shape_of_theta[0]
    number_of_columns_of_theta = shape_of_theta[1]
    if (number_of_rows_of_theta != number_of_columns_of_x) & (number_of_columns_of_theta != 1):
        raise ValueError(error.MATRIX_DIMENSION_MISMATCH_ERROR)
    if (number_of_rows_of_x != number_of_rows_of_y) & (number_of_columns_of_y != 1):
        raise ValueError(error.MATRIX_DIMENSION_MISMATCH_ERROR)


def validate_matrix_multiplication_dimensions(x, y):
    """
    This function validates whether two matrices to be multiplied satisfy matrix
    multiplication condition

    :param x:
    :param y:
    :return:
    """
    shape_of_x = numpy.shape(x)
    number_of_columns_of_x = shape_of_x[1]
    shape_of_y = numpy.shape(y)
    number_of_rows_of_y = shape_of_y[0]
    if number_of_columns_of_x != number_of_rows_of_y:
        raise ValueError(error.MATRIX_DIMENSION_MISMATCH_ERROR)
