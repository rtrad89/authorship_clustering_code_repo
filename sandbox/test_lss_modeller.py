# -*- coding: utf-8 -*-
"""
Unittest for lss_modeller controller

@author: RTRAD
"""
import unittest
import lss_modeller
import os


class TestLssModeller(unittest.TestCase):

    # Define class level routines
    @classmethod
    def setUpClass(cls):
        cls.test_path = r".\test_data"
        # Create the output directory
        if not os.path.exists(cls.test_path):
            os.makedirs(cls.test_path)

    @classmethod
    def tearDownClass(cls):
        # Delete the test outputs to set up a new slate for the next tests
        if os.path.exists(cls.test_path):
            os.shutil.rmtree(cls.test_path)

    # Define pre and post test routines
    def setUp(self):
        pass

    def tearDown(self):
        pass
