import numpy as np


class Datasets(object):
    def __init__(self, df):
        """

        :param df: pandas dataframe containing ratings. The dataframe is composed of rows with
        posts as indices, columns as columns and entries with ratings that a user gives to an item.
        """
        self.df = df

    @property
    def review_matrix(self):
        """
        Convert the dataframe to a numpy array.
        """
        return self.df.to_numpy()

    @property
    def users(self):
        """
        `user_id`s in the columns participating in the rating matrix.
        """
        return self.df.columns.values

    @property
    def posts(self):
        """
        `post_id`s in the indices participating in the rating matrix.
        """
        return self.df.index.values

    def train_test_split(self, test_size):
        """
        Train-test split method. We create a mask array with the same shape as the original
        numpy array, with True and False entries with probabilities given by 1- test_size, test_size,
        respectively. We then apply this mask and its inverse to the original ratings array in order
        to produce the train and test sets. This implies that the proportion of samples in the test size
        will not be exactly test_size.
        :param test_size: fraction of the total data set dedicated to the test set.
        :return: train and test numpy arrays with the same shape as the original dataframe.
        """
        train_matrix = self.review_matrix.copy()
        test_matrix = self.review_matrix.copy()

        mask = ~np.isnan(self.review_matrix)

        n_posts, n_users = mask.shape
        for post_idx in range(n_posts):
            for user_idx in range(n_users):
                if mask[post_idx, user_idx]:
                    take_it_to_test = np.random.choice(a=[False, True], p=[1 - test_size, test_size])
                    if take_it_to_test:
                        train_matrix[post_idx, user_idx] = np.nan
                    else:
                        test_matrix[post_idx, user_idx] = np.nan
        return train_matrix, test_matrix
