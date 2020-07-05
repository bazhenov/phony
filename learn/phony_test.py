import unittest
import phony

class TestStringMethods(unittest.TestCase):

  def test_2vec(self):
    (x, y) = phony.sample2vec("123", [[0, 1], [2, 3]])
    self.assertEqual(x, [49, 50, 51])
    self.assertEqual(y, [1, 0, 1])

  def test_2tensor(self):
    (x, y) = phony.vec2tensor(([49, 50, 51], [1, 0, 1]), l=7, idx=0)
    self.assertEqual(x, [[0, 0, 0, 49, 50, 51, 0], [0, 0, 0, 0, 0, 1, 0]])
    self.assertEqual(y, 1)

if __name__ == '__main__':
    unittest.main()