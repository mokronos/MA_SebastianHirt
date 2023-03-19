"""Module providing testing"""
import unittest
from src import helpers
import numpy as np
from src import exp

class TestMetric(unittest.TestCase):
    '''
    Test metrics
    '''

    def test_text_error_rate(self):
        '''
        Test text error rate
        '''

        string1 = 'this is a test 2321'
        string2 = 'this is a test 1321!'

        with self.subTest(string1=string1, string2=string2):
            self.assertEqual(helpers.text_error_rate(string1, string2), 0.1)

        string1 = 'test         rs2'
        string2 = 'test rs2'

        with self.subTest(string1=string1, string2=string2):
            self.assertEqual(helpers.text_error_rate(string1, string2), 0.5)

    def test_nonlinearfitting(self):
        '''
        Test nonlinearfitting
        '''

        # put values from wiki for spearman correlation into matlab code
        # https://github.com/lllllllllllll-llll/SROCC_PLCC_calculate
        # https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        # and made the results (srocc, plcc, ypre) the test case, srocc value
        # fits the value on wiki
        objvals = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]
        subjvals = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]

        ypre_gt = [21.7778, 21.7778,  2.0000, 21.7778, 21.7778,
                   21.7778, 21.7778, 21.7778, 21.7778, 21.7778]
        srocc_gt = -0.175758
        plcc_gt = 0.435569

        srocc, plcc, ypre = exp.nonlinearfitting(objvals, subjvals)

        self.assertTrue(np.isclose(srocc, srocc_gt))
        self.assertTrue(np.isclose(plcc, plcc_gt))
        self.assertTrue(np.isclose(ypre, ypre_gt, rtol=0.002).all())
