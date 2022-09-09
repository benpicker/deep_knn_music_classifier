import unittest
from core.creating_features_and_preprocessing.create_raw_features import get_features
from utils import get_project_root


class CreateRawFeaturesTests(unittest.TestCase):
    def test_get_features(self):
        root_directory = get_project_root()
        genre = "blues"
        file_name = f"{genre}.00000.au"
        sample_file_path = f"{root_directory}/data/genres/blues/blues.00000.au"
        features = get_features(sample_file_path)
        expected_features = ['0.34995052218437195', '0.13019199669361115', '1784.4165459703192',
                             '2002.6571063943213', '3806.4186497738488', '0.08306639113293343', '-113.619384765625',
                             '121.55303192138672', '-19.15105628967285', '42.34576416015625', '-6.371164798736572',
                             '18.61302947998047', '-13.692057609558105', '15.339376449584961', '-12.283616065979004',
                             '10.973773002624512', '-8.322407722473145', '8.806784629821777', '-3.665800094604492',
                             '5.745936393737793', '-5.161709308624268', '0.7502961158752441', '-1.6883563995361328',
                             '-0.4093300998210907', '-2.298867702484131', '1.2199459075927734']
        self.assertEqual(expected_features, features)

if __name__ == '__main__':
    unittest.main()
