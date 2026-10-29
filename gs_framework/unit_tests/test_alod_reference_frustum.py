import unittest

import numpy as np

from alod_reference import extract_four_frustum_planes, frustum_visible_spheres


class AlodReferenceFrustumTest(unittest.TestCase):
    def setUp(self):
        self.planes = extract_four_frustum_planes(np.eye(4, dtype=np.float32))

    def _visible(self, point, radius=0.0):
        return bool(
            frustum_visible_spheres(
                np.asarray([point], dtype=np.float32),
                np.asarray([radius], dtype=np.float32),
                self.planes,
            )[0]
        )

    def test_extracts_only_left_right_bottom_top_in_source_order(self):
        expected = np.asarray(
            [[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(self.planes, expected)
        self.assertEqual(self.planes.shape, (4, 4))

    def test_inside_and_each_outside_plane(self):
        self.assertTrue(self._visible([0, 0, 1000]))
        self.assertFalse(self._visible([-2, 0, 0]))
        self.assertFalse(self._visible([2, 0, 0]))
        self.assertFalse(self._visible([0, -2, 0]))
        self.assertFalse(self._visible([0, 2, 0]))

    def test_plane_contact_is_visible_without_epsilon(self):
        self.assertTrue(self._visible([-1, 0, 0]))
        self.assertFalse(self._visible([np.nextafter(np.float32(-1), np.float32(-2)), 0, 0]))
        self.assertTrue(self._visible([-1.1, 0, 0], radius=0.2))


if __name__ == "__main__":
    unittest.main()
