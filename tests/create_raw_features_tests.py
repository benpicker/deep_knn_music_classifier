import unittest
from core.creating_features_and_preprocessing.create_raw_features import get_features
from utils import get_project_root
import numpy as np

class CreateRawFeaturesTests(unittest.TestCase):
    def test_get_features(self):
        root_directory = get_project_root()
        genre = "blues"
        file_name = f"{genre}.00000.au"
        sample_file_path = f"{root_directory}/data/genres/blues/blues.00000.au"
        features = get_features(sample_file_path)
        expected_features = np.array([ 3.49950522e-01,  1.30191997e-01,  1.78441655e+03,  2.00265711e+03,
                                       3.80641865e+03,  8.30663911e-02, -1.13619385e+02,  1.21553032e+02,
                                       -1.91510563e+01,  4.23457642e+01, -6.37116480e+00,  1.86130295e+01,
                                       -1.36920576e+01,  1.53393764e+01, -1.22836161e+01,  1.09737730e+01,
                                       -8.32240772e+00,  8.80678463e+00, -3.66580009e+00,  5.74593639e+00,
                                       -5.16170931e+00,  7.50296116e-01, -1.68835640e+00, -4.09330100e-01,
                                       -2.29886770e+00,  1.21994591e+00])
        for i in range(len(expected_features)):
            self.assertAlmostEqual(expected_features[i], features[i],places=5)

if __name__ == '__main__':
    unittest.main()
