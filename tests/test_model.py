import os
import random

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from afinidata_recommender.recommender.models import CollaborativeFiltering
from afinidata_recommender.recommender.read_db import ReadDatabase
from afinidata_recommender.recommender.preprocess import SetUpDataframes
from afinidata_recommender.recommender.datasets import Datasets


#@pytest.fixture
#def response_df():
#    return pd.read_csv('response_df.csv')


@pytest.fixture
def random_response_df():
    data = {
        'user_id': np.arange(1, 101),
        'response': np.random.randint(1, 4, 100),
        'question_id': np.arange(1, 101)
    }
    return pd.DataFrame(data=data)


@pytest.fixture
def zero_response_matrix():
    return np.zeros((100, 100))


@pytest.fixture
def random_response_matrix():
    return np.random.rand(100, 100)


@pytest.fixture
def zero_model():
    model = CollaborativeFiltering()
    model.n_users = 100
    model.n_items = 100

    n_features = 2

    model.parameters['mean_rating'] = np.zeros((1, 1))
    model.parameters['bias_user'] = np.zeros((1, model.n_users))
    model.parameters['bias_item'] = np.zeros((model.n_items, 1))
    model.parameters['feat_vec'] = np.zeros((n_features, model.n_items))
    model.parameters['user_vec'] = np.zeros((n_features, model.n_users))

    return model


@pytest.fixture
def random_model():
    model = CollaborativeFiltering()
    model.n_users = 100
    model.n_items = 100

    n_features = 2
    alpha = 0.

    model._initialize_model(n_features, alpha)
    return model


class TestTraining:
    def test_loss_function(self, zero_response_matrix, zero_model):
        alpha = 0
        assert zero_model.loss(
            review_matrix=zero_response_matrix,
            mu=zero_model.parameters['mean_rating'],
            b_user=zero_model.parameters['bias_user'],
            b_item=zero_model.parameters['bias_item'],
            x=zero_model.parameters['feat_vec'],
            theta=zero_model.parameters['user_vec'],
            alpha=alpha
        ) <= 1e-09

    def test_gradients(self, random_response_matrix, random_model):
        """
        This test implements the definition of a gradient and compares its result with the result for
        gradients from the parameter_gradients method.
        """
        alpha = 0.
        n_features = 2
        lr = 0.00000001

        mu = random_model.parameters['mean_rating']
        b_user = random_model.parameters['bias_user']
        b_item = random_model.parameters['bias_item']
        x = random_model.parameters['feat_vec']
        theta = random_model.parameters['user_vec']

        mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad = random_model.parameter_gradients(
            review_matrix=random_response_matrix,
            mu=mu,
            b_user=b_user,
            b_item=b_item,
            x=x,
            theta=theta,
            alpha=alpha,
            n_features=n_features
        )

        mu_adv = mu + lr

        b_user_adv = b_user.copy()
        b_user_adv[0, 5] += lr

        b_item_adv = b_item.copy()
        b_item_adv[5, 0] += lr

        mu_grad_def = (random_model.loss(random_response_matrix, mu_adv, b_user, b_item, x, theta, alpha) -
                       random_model.loss(random_response_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_user_grad_def = (random_model.loss(random_response_matrix, mu, b_user_adv, b_item, x, theta, alpha) -
                           random_model.loss(random_response_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_item_grad_def = (random_model.loss(random_response_matrix, mu, b_user, b_item_adv, x, theta, alpha) -
                           random_model.loss(random_response_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        assert np.abs(mu_grad - mu_grad_def) <= 1e-04
        assert np.abs(b_user_grad[0, 5] - b_user_grad_def) <= 1e-04
        assert np.abs(b_item_grad[5, 0] - b_item_grad_def) <= 1e-04

    def test_training(self, random_response_df, random_model):
        response_matrix = SetUpDataframes.response_matrix(random_response_df)

        # train test split
        datasets = Datasets(response_matrix)
        train_set, test_set = datasets.train_test_split(0.1)

        # model initialization
        random_model.actors = {
            'users': response_matrix.columns.values,
            'items': response_matrix.index.values
        }

        parameters_before = {parameter: values for parameter, values in random_model.parameters.items()}

        random_model.train(
            train_matrix=train_set,
            test_matrix=test_set,
            epochs=1,
            alpha=0.,
            n_features=2,
            lr=0.01,
            resume=False
        )

        parameters_after = {parameter: values for parameter, values in random_model.parameters.items()}

        for parameter in parameters_before:
            assert np.linalg.norm(parameters_before[parameter] - parameters_after[parameter]) > 0.0

        assert random_model.has_been_trained

    def test_prediction(self, zero_model):
        model_predictions = zero_model.predict_rating(50)['predictions']
        model_predictions_default = zero_model.predict_rating(1000)['predictions']

        assert model_predictions.mean() == 0
        assert model_predictions_default.mean() == 0


class TestReadDatabase:

    def test_read_max_age(self):
        load_dotenv('.env')

        # environment variables
        DB_URI = os.environ.get("DB_URI")

        # set up database reader
        engine = create_engine(DB_URI)
        reader_cm = ReadDatabase(engine, 'CM_BD')

        max_ages = reader_cm.get_data(
            'id, max_range',
            'posts_post'
        )

        max_age = max_ages['max_range'].max()
        assert max_age == 200

        min_ages = reader_cm.get_data(
            'id, min_range',
            'posts_post',
            "status IN ('published')"
        )

        min_age = min_ages['min_range'].min()
        assert min_age == 0


