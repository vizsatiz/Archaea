import machine_learning.linear_regression.linear_regression as lr
import unittest
import numpy

alpha = 0
no_of_iterations = 0
theta = 0.001
lr_object = lr.LinearRegression(alpha, no_of_iterations, theta)


class TestLinearRegression(unittest.TestCase):
    """
    This test checks the zero error condition for cost function.
    """

    def test_cost_function_zero_error(self):
        x = numpy.matrix('1 2;3 4')
        theta = numpy.matrix('7;8')
        y = numpy.matrix('23;53')
        cost_value = lr_object.cost(x, y, theta)
        self.assertEqual(cost_value, 0)

    def test_cost_function_non_zero_error(self):
        x = numpy.matrix('1 2;3 4')
        theta = numpy.matrix('7;8')
        y = numpy.matrix('23;52')
        cost_value = lr_object.cost(x, y, theta)
        print cost_value
        #self.assertEqual(cost_value, 0)


if __name__ == '__main__':
    unittest.main()
