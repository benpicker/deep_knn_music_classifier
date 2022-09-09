import unittest
import torch
from utils import GTZANDerivedDataDataset, get_project_root

class utilts_tests(unittest.TestCase):
    def setUp(self) -> None:
        root_directory = get_project_root()
        data_file_directory = f"{root_directory}/data/feature_data/"
        genre_label_file = f"{data_file_directory}/y_labels.csv"
        audio_data_file_path = f"{data_file_directory}/X_features.csv"
        self.sample_dataset = GTZANDerivedDataDataset(genre_label_file, audio_data_file_path)

    def test_GTZANDerivedDataDataset(self):
        expected = ["0 - blues", "1 - classical", "2 - country", "3 - disco", "4 - hiphop", "5 - jazz",
                        "6 - metal", "7 - pop", "8 - reggae", "9 - rock", ]
        self.assertEqual(expected,self.sample_dataset.classes)
        expected = {'0 - blues': 0, '1 - classical': 1, '2 - country': 2, '3 - disco': 3, '4 - hiphop': 4,
                    '5 - jazz': 5, '6 - metal': 6, '7 - pop': 7, '8 - reggae': 8, '9 - rock': 9}
        self.assertEqual(expected,self.sample_dataset.class_to_idx)
        # validate data has expected dimensions
        self.assertEqual([1000,26], list(self.sample_dataset.data.shape))
        # validate data has expected dimensions
        self.assertEqual([1000], list(self.sample_dataset.targets.shape))

        # validate __len__ works
        self.assertEqual(1000, len(self.sample_dataset))

        # validate __getitem__
        x, y = self.sample_dataset.__getitem__(0)

        expected_x = torch.tensor([-0.8848, -0.2658, -1.4613, -1.4159, -1.5257, -1.0768, -0.8920,  1.1695,
                                   0.5625,  0.7587,  0.8051, -0.0554,  1.0554, -0.9302, -0.7858,  1.1500,
                                   1.6366, -0.9984,  1.4206,  0.7062,  0.9614, -0.4907, -0.1256, -0.7606,
                                   1.5034,  0.1253], dtype=torch.float64)
        # same numbers
        for i in range(len(x)):
            self.assertAlmostEqual(expected_x[i].item(), x[i].item(), places=3)

        # same type
        self.assertEqual(type(expected_x), type(x))
        # target -- same value
        self.assertEqual(0, y.item())
        # target -- same type
        self.assertEqual(type(0), type(y.item()))

        # validate it splits into training sets
        train_size = int(0.7 * len(self.sample_dataset))
        test_size = len(self.sample_dataset) - train_size
        train,test = torch.utils.data.random_split(self.sample_dataset, [train_size, test_size])
        self.assertEqual(700, len(train))
        self.assertEqual(300, len(test))

if __name__ == '__main__':
    unittest.main()