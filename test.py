#!/usr/bin/env python

import unittest
from phony import Augmentator

class AugmentatorTest(unittest.TestCase):

  def test_format(self):
    ag = Augmentator()
    out = ag.format((7, 4232, 243451), format=('+7', '(####)', '###-###'), whitespaces=True)
    self.assertEqual(out, "+7 (4232) 243-451")

  def test_eight_rule(self):
    self.assertEqual(Augmentator.eight_rule("79141112233"), "89141112233")
    self.assertEqual(Augmentator.eight_rule("7 (914) 1112233"), "8 (914) 1112233")
    self.assertEqual(Augmentator.eight_rule("+79141112233"), "89141112233")

  def test_random_punctuation_rule(self):
    self.assertEqual(Augmentator.random_punctuation_rule("7914", character="/", position=1, count=2), "7//914")

  def test_qasinumber_rule(self):
    self.assertEqual(Augmentator.quasinumber_rule("41"), "Чl")

  def test_augmentator_assembly(self):
    a = Augmentator()
    a.setup_rules()
    out = a.format((7, 914, 7057823))

if __name__ == '__main__':
    unittest.main()
