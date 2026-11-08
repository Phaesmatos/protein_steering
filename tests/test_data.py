import unittest
from steer import make_collagen, AA

class TestSyntheticData(unittest.TestCase):
    def test_motifs_and_composition(self):
        examples = make_collagen(20)
        for pos, neg in examples:
            self.assertEqual(len(pos), len(neg))
            self.assertEqual(sorted(pos), sorted(neg))
            self.assertTrue(all(pos[i] == 'G' for i in range(0, len(pos), 3)))
            self.assertEqual(neg, pos[1:] + pos[:1])
            self.assertTrue(set(pos).issubset(set(AA)))

if __name__ == '__main__':
    unittest.main()
