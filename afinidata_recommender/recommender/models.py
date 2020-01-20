import numpy as np
import pandas as pd
import pickle


class CollaborativeFiltering(object):
    """
    Collaborative filtering recommender system.
    """

    parameters = {}
    losses = {}
    hyperparameters = {}
    has_been_trained = False
    actors = {}
    n_users = 0
    n_items = 0

    def __init__(self):
        self.name = 'Collaborative Filtering'

    def _initialize_model(self, n_features, alpha):
        """
        Initialize parameters to random numbers, losses to empty arrays and hyperparameters to
        given arguments.
        :param n_features: integer, number of latent features for the collaborative filtering method.
        :param alpha: float, weight for the l2 regularization term in the model, the larger this number,
        the strongest the regularization effect.
        """
        self.parameters = {
            'mean_rating': 0.001 * np.random.rand(1, 1),
            'bias_user': 0.001 * np.random.rand(1, self.n_users),
            'bias_item': 0.001 * np.random.rand(self.n_items, 1),
            'feat_vec': 0.001 * np.random.rand(n_features, self.n_items),
            'user_vec': 0.001 * np.random.rand(n_features, self.n_users)
        }
        self.losses = {
            'train': np.array([]),
            'test': np.array([])
        }
        self.hyperparameters = {
            'regularization': alpha,
            'latent features': n_features
        }

    def _loss_root(self, review_matrix, mu, b_user, b_item, x, theta):
        return mu + b_user + b_item + np.dot(x.T, theta) - review_matrix

    def loss(self, review_matrix, mu, b_user, b_item, x, theta, alpha):
        """
        Loss function with regularization.
        :param review_matrix: numpy array with shape (n_items, n_users) containing ratings.
        :param mu: model parameter with shape (1, 1) for the mean rating over all users and items.
        :param b_user: model parameter with shape (1, n_users) for the rating bias for each user.
        :param b_item: model parameter with shape (n_items, 1) for the rating bias for each item.
        :param x: model parameter with shape (n_features, n_items) for the latent item features
        in the matrix factorization involved in collaborative filtering.
        :param theta: model parameter with shape (n_features, n_users) for the latent user features
        in the matrix factorization involved in collaborative filtering.
        :param alpha: float, regularization weight.
        :return: float, l2 loss function with l2 regularization, given the model parameters.
        """
        return (1 / 2.) * np.nansum(np.square(self._loss_root(review_matrix, mu, b_user, b_item, x, theta))) \
               + (alpha / 2.) * np.sum(np.square(x)) \
               + (alpha / 2.) * np.sum(np.square(theta)) \
               + (alpha / 2.) * np.sum(np.square(b_item)) \
               + (alpha / 2.) * np.sum(np.square(b_user))

    @staticmethod
    def predict(mu, b_user, b_item, x, theta):
        """
        Predict and reconstruct the review matrix.
        :param mu: model parameter with shape (1, 1) for the mean rating over all users and items.
        :param b_user: model parameter with shape (1, n_users) for the rating bias for each user.
        :param b_item: model parameter with shape (n_items, 1) for the rating bias for each item.
        :param x: model parameter with shape (n_features, n_items) for the latent item features
        in the matrix factorization involved in collaborative filtering.
        :param theta: model parameter with shape (n_features, n_users) for the latent user features
        in the matrix factorization involved in collaborative filtering.
        :return: numpy array with shape (n_items, n_users) with the predicted ratings.
        """
        return mu + b_user + b_item + np.dot(x.T, theta)

    @staticmethod
    def predict_default(mu, b_item):
        """
        Predictions for users that have no ratings. The model becomes a popularity model where the prediction
        for a particular item corresponds to mean + item bias.
        :param mu: model parameter with shape (1, 1) for the mean rating over all users and items.
        :param b_item: model parameter with shape (n_items, 1) for the rating bias for each item.
        :return: numpy array with shape (n_items, n_users) with the predicted ratings according to popularity.
        """
        return mu + b_item

    def verify_gradients(self, review_matrix, alpha, n_features, lr):
        """
        This method implements the definition of a gradient and compares its result with the result for
        gradients from the parameter_gradients method.
        """
        mu = 0.001 * np.random.rand(1, 1)
        b_user = 0.001 * np.random.rand(1, self.n_users)
        b_item = 0.001 * np.random.rand(self.n_items, 1)
        x = 0.001 * np.random.rand(n_features, self.n_items)
        theta = 0.001 * np.random.rand(n_features, self.n_users)

        mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad = self.parameter_gradients(
            review_matrix, mu, b_user, b_item, x, theta, alpha, n_features
        )

        mu_adv = mu + lr

        b_user_adv = b_user.copy()
        b_user_adv[0, 5] += lr

        b_item_adv = b_item.copy()
        b_item_adv[5, 0] += lr

        mu_grad_def = (self.loss(review_matrix, mu_adv, b_user, b_item, x, theta, alpha) -
                       self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_user_grad_def = (self.loss(review_matrix, mu, b_user_adv, b_item, x, theta, alpha) -
                           self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        b_item_grad_def = (self.loss(review_matrix, mu, b_user, b_item_adv, x, theta, alpha) -
                           self.loss(review_matrix, mu, b_user, b_item, x, theta, alpha)) / lr

        print(f'Gradient method: mu     {mu_grad:.5f} / gradient definition {mu_grad_def:.5f} / difference {mu_grad - mu_grad_def}')
        print(f'Gradient method: b_user {b_user_grad[0,5]:.5f} / gradient definition {b_user_grad_def:.5f} / difference {b_user_grad[0,5] - b_user_grad_def}')
        print(f'Gradient method: b_item {b_item_grad[5,0]:.5f} / gradient definition {b_item_grad_def:.5f} / difference {b_item_grad[5,0] - b_item_grad_def}')

    @staticmethod
    def parameter_gradients(review_matrix, mu, b_user, b_item, x, theta, alpha, n_features):
        """
        Gradients for the loss function with respect to all model parameters, according to the analytical
        expressions for the loss function.
        :param review_matrix: numpy array with shape (n_items, n_users) containing ratings.
        :param mu: model parameter with shape (1, 1) for the mean rating over all users and items.
        :param b_user: model parameter with shape (1, n_users) for the rating bias for each user.
        :param b_item: model parameter with shape (n_items, 1) for the rating bias for each item.
        :param x: model parameter with shape (n_features, n_items) for the latent item features
        in the matrix factorization involved in collaborative filtering.
        :param theta: model parameter with shape (n_features, n_users) for the latent user features
        in the matrix factorization involved in collaborative filtering.
        :param alpha: float, regularization weight.
        :param n_features: latent features for the depth of matrix factorization
        :return: tuple containing numpy arrays for gradients with respect to all parameters.
        """
        mask = ~np.isnan(review_matrix)
        n_posts, n_users = review_matrix.shape

        x_grad = np.zeros(x.shape)
        theta_grad = np.zeros(theta.shape)

        loss_root = mu + b_item + b_user + np.dot(x.T, theta) - review_matrix

        mu_grad = np.nansum(loss_root)

        b_user_grad = np.nansum(loss_root, axis=0).reshape(1, -1) + alpha * b_user

        b_item_grad = np.nansum(loss_root, axis=1).reshape(-1, 1) + alpha * b_item

        # run over posts
        for post in range(n_posts):
            # select users that reviewed post
            idx = mask[post, :]
            # restrict Y and Theta to users who reviewed post
            review_matrix_temp = review_matrix[post, idx].reshape(1, -1)
            theta_temp = theta[:, idx].reshape(n_features, -1)
            x_temp = x[:, post].reshape(n_features, -1)
            x_grad[:, post] = np.dot(theta_temp, np.dot(theta_temp.T, x_temp) - review_matrix_temp.T)\
                .reshape(n_features)
        x_grad = x_grad + alpha * x

        # run over users
        for user in range(n_users):
            idx = mask[:, user]
            # restrict Y, X  and Theta to posts reviewed by user
            review_matrix_temp = review_matrix[idx, user].reshape(-1, 1)
            theta_temp = theta[:, user].reshape(n_features, -1)
            x_temp = x[:, idx].reshape(n_features, -1)
            theta_grad[:, user] = np.dot(x_temp, np.dot(x_temp.T, theta_temp) - review_matrix_temp)\
                .reshape(n_features)
        theta_grad = theta_grad + alpha * theta

        return mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad

    def train(self, resume, train_matrix, test_matrix, epochs, alpha, n_features, lr):
        """
        Training phase in which the loss function is minimized in a sequence of minimization steps
        corresponding to the gradient descent method using the entire training dataset. After each
        minimization step, the loss function is computed for the train and test data. The result of
        this method is a collection of model parameters, stored as class properties, which minimize
        the loss function.
        :param resume: boolean, if True, model parameters are loaded from the class properties and
        training resumes starting at those parameters; if False, model parameters are randomly
        initialized and minimization starts from those parameters.
        :param train_matrix: numpy array with shape (n_items, n_users) corresponding to the train
        rating matrix.
        :param test_matrix: numpy array with shape (n_items, n_users) corresponding to the test
        rating matrix.
        :param epochs: integer, number of training epochs.
        :param alpha: float, regularization weight.
        :param n_features: integer, latent features.
        :param lr: float, step size for the gradient descent method.
        """
        if not resume:
            self._initialize_model(n_features, alpha)
            self.has_been_trained = True
        else:
            assert self.has_been_trained, 'The model has not been trained or loaded'
            alpha = self.hyperparameters['regularization']
            n_features = self.hyperparameters['latent features']

        n_train = np.count_nonzero(~np.isnan(train_matrix))
        n_test = np.count_nonzero(~np.isnan(test_matrix))

        print('*' * 80)
        print(f'training recommendation model for {epochs} epochs with learning rate {lr} and \n' +
              f'hyperparameters regularization: {alpha} / latent features: {n_features}')
        print('*' * 80)
        for epoch in range(epochs):
            mu_grad, b_user_grad, b_item_grad, x_grad, theta_grad = self.parameter_gradients(
                review_matrix=train_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=alpha,
                n_features=n_features
            )

            self.parameters['mean_rating'] -= lr * mu_grad
            self.parameters['bias_user'] -= lr * b_user_grad
            self.parameters['bias_item'] -= lr * b_item_grad
            self.parameters['feat_vec'] -= lr * x_grad
            self.parameters['user_vec'] -= lr * theta_grad

            train_loss = self.loss(
                review_matrix=train_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=alpha) / n_train

            self.losses['train'] = np.append(self.losses['train'], train_loss)

            test_loss = self.loss(
                review_matrix=test_matrix,
                mu=self.parameters['mean_rating'],
                b_user=self.parameters['bias_user'],
                b_item=self.parameters['bias_item'],
                x=self.parameters['feat_vec'],
                theta=self.parameters['user_vec'],
                alpha=0) / n_test
            self.losses['test'] = np.append(self.losses['test'], test_loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1:05d} / train loss {train_loss:.6f} / test loss {test_loss:.6f}')
        print('*' * 80)
        print('training finished. final losses are')
        print(f'Epoch {epochs:05d} / train loss {train_loss:.6f} / test loss {test_loss:.6f}')

    def predict_rating(self, idx):
        """
         Creates a pandas dataframe with the predicted ratings for a particular user specified by its
         index in the numpy rating matrix. The resulting dataframe must then be related to a particular
         user by finding the user_id corresponding to that particular array index. If the idx does not
         exist in the predictions array, the predictions corresponds to a popularity based model.
         :param idx: numpy array column index.
         :return: pandas dataframe with column `predictions`.
        """
        try:
            predictions = self.predict(
                self.parameters['mean_rating'],
                self.parameters['bias_user'],
                self.parameters['bias_item'],
                self.parameters['feat_vec'],
                self.parameters['user_vec']
                )[:, idx]

        except IndexError:
            predictions = self.predict_default(
                self.parameters['mean_rating'],
                self.parameters['bias_item']
            )[:, 0]

        n_items = predictions.shape[0]

        predictions_with_id = np.zeros((n_items, 2))
        predictions_with_id[:, 1] = predictions

        return pd.DataFrame({
            'predictions': predictions_with_id[:, 1]
        })

    def save_model(self, filename):
        """
        Creates a specification for a collaborative filtering model and saves it to a pickle file. The
        specification includes model hyperparameters, model parameters, test and train losses and lists
        of users and items.
        :param filename: string, filename without extension, to be saved with respect to the script
        directory.
        """
        model_specs = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'hyperparameters': self.hyperparameters,
            'parameters': self.parameters,
            'losses': self.losses,
            'actors': self.actors
        }
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(model_specs, f)

    def load_model(self, filename):
        """
        Loads the pickle file with filename specified containing the model specifications into the class
        properties.
        :param filename: string, filename without extensions, specified with respect to the script directory.
        """
        with open(f'{filename}.pkl', 'rb') as f:
            model_specs = pickle.load(f)
            self.n_users = model_specs['n_users']
            self.n_items = model_specs['n_items']
            self.hyperparameters = model_specs['hyperparameters']
            self.parameters = model_specs['parameters']
            self.losses = model_specs['losses']
            self.actors = model_specs['actors']
            self.has_been_trained = True

    def afinidata_recommend(self, user_id, months, question_df, taxonomy_df, content_df, response_df, sent_activities):
        """
        Creates a pandas dataframe containing the ranked items for a specific user. This method implements
        the specific decisions made with respect to the role the collaborative filtering model must play within
        the way we consider content should be delivered.

        The sequence of decisions is the following:
        1. We get the predictions for a particular user,
        2. The content is divided into several categories. We choose most probably the category in which
        the user has performed worst in average, according to empirical data. If there is not available
        data for a particular category, we use the average predicted rating for such category. This category
        selection is non-deterministic, that is, we create some probability distribution over categories
        that priviledges a category as its mean rating decreases. This randomness allows for variation.
        3. Once we select a category, we choose the activity with the highest predicted rating.

        We apply the following filters: we remove activities that have already been seen, although not
        necessarily rated; we select only activities from the age range to which the user belongs. If, after
        these filters, there are no activities for a category, these category is removed from the probability
        distribution. If there are no activities in all categories, the filter for seen activities is not
        applied.
        :param user_id: integer.
        :param months: integer, age of the user in months.
        :param question_df: pandas dataframe with the questions information (question_id, post_id).
        :param taxonomy_df: pandas dataframe with the activities categories, to which category belongs each
        activity.
        :param content_df: pandas dataframe with content information (post_id, age ranges).
        :param response_df: pandas dataframe with the response information (response, response_id, question_id).
        :param sent_activities: list of activities already sent to the user.
        :return: pandas dataframe containing the ranked activities from the selected category.
        """
        # data is sequentially ordered and the relation between the indices and the actual
        # user_id is stored in self.actors['users']. if the user is in this list, which means
        # that this user has given at least one rating, then find it, else go to the
        # exceptional case handled by self.predict_rating.
        if user_id in self.actors['users']:
            idx, = np.where(self.actors['users'] == user_id)
            predictions = self.predict_rating(idx[0])
        else:
            predictions = self.predict_rating(-1)
        predictions['question_id'] = self.actors['items']

        response_df = response_df[response_df['user_id'] == user_id]

        # we add the columns corresponding to question_id and area_id
        predictions = pd.merge(predictions, question_df, 'inner', left_on='question_id', right_on='id')
        predictions = pd.merge(predictions, taxonomy_df, 'inner', 'post_id')
        predictions = pd.merge(predictions, response_df, 'outer', 'question_id')
        predictions.drop(['id', 'user_id'], axis=1, inplace=True)

        # lists from which we are going to filter next, we will only deliver content appropiate
        # for the age and activities not sent
        content_for_age = content_df[(content_df['min_range'] <= months) & (content_df['max_range'] >= months)][
            'id'].values.tolist()

        relevant_predictions = predictions[predictions['post_id'].isin(content_for_age)]
        relevant_unseen_predictions = relevant_predictions[~relevant_predictions['post_id'].isin(sent_activities)]

        area_performance = relevant_predictions[['predictions', 'area_id', 'response']].groupby('area_id').apply(
            lambda g: g.mean(skipna=True))
        area_performance['score'] = area_performance[
            ['response', 'predictions']].apply(
            lambda row: row['predictions'] if pd.isna(row['response']) else row['response'], axis=1)

        # we normalize the mean predictions by area
        area_performance['normalized'] = area_performance['score'].apply(
            lambda x: (x - area_performance['score'].mean()) / (0.001 + area_performance['score'].std()))
        # we compute probabilities from the normalized means such that lower means correspond
        # to higher probabilities
        area_performance['probabilities'] = area_performance['normalized'].apply(lambda x: np.exp(-x))

        # the logic for selecting an area is the following. if after applying the seen activities and
        # age content, there are no activities left, send an activity from the pool of all
        # activities appropiate for an age that has already been seen. if, after these filters,
        # there is something left, select from the areas for which there are activities left.
        if len(relevant_unseen_predictions.index) == 0:
            predictions_temp = relevant_predictions
        else:
            available_areas = relevant_unseen_predictions['area_id'].unique()
            area_performance = area_performance[area_performance.index.isin(available_areas)]
            predictions_temp = relevant_unseen_predictions

        area_performance['probabilities'] = area_performance['probabilities'] / area_performance['probabilities'].sum()

        print(area_performance)

        # we randomly select an area according to the assigned probabilities
        selected_area = np.random.choice(area_performance.index.values, p=area_performance['probabilities'].values)
        return predictions_temp[
            predictions_temp['area_id'] == selected_area
            ].sort_values('predictions', ascending=False)
