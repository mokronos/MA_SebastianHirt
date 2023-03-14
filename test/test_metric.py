"""Module providing testing"""
import unittest
from src import helpers


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
