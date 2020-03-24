[![codecov](https://codecov.io/gh/afinidata2019/afinidata_recommender/branch/master/graph/badge.svg)](https://codecov.io/gh/afinidata2019/afinidata_recommender)
[![Documentation Status](https://readthedocs.org/projects/afinidata-recommender/badge/?version=latest)](https://afinidata-recommender.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/afinidata2019/afinidata_recommender.svg?branch=master)](https://travis-ci.com/afinidata2019/afinidata_recommender)

# **afinidata** recommender

## About

This is **afinidata**'s recommendation engine for activities that children can perform. The model is based on child's performance on past activities.

## Details

We have developed an *ad hoc* recommendation engine that delivers activities implementing our approach to how children should best develop their skills. At the core of the recommendation engine lies a collaborative filtering model, with explicitly parameterized mean and user and item biases.

Our training data consists of the reports that **afinidata** users have made of their children's performance, that is, our review matrix for collaborative filtering has as entries *r(u, i)* the performance that user *u* has reported for her child on activity *i*.

The model is collaborative filtering with matrix factorization and explicitly parameterized overall mean and and user and item biases. The model's parameters are the overall mean, the user bias vector, the item bias vector and the matrices in which the remaining modeled behavior factorizes. The size of these factor matrices defines the number of latent features and is a hyperparameter of the model. The model is trained using gradient descent over the entire training set using an quadratic loss function and a regularization term, whose weight is the second hyperparameter of the model. We choose hyperparameters using standard validation techniques over a set of hyperparameters. The model is implemented in **python** from scratch using standard numerical libraries.

Each of the activities belongs to one of three child development areas: cognitive, motor and emotional. The way we have chosen to use our performance prediction model is the following: we recommend the activity in which we predict the child is going to perform best from the area in which she has empirically performed worst in average. This way we provide content that does not specialize the child and that makes her improve in less developed areas. The development area is also chosen randomly according to a probability distribution that precisely favors areas with lowest performance, so that there is some variability in the result of our recommendation engine.

## Running

We recommend that an environmente be created:

    $ git clone https://github.com/afinidata2019/afinidata_recommender.git
    $ cd afinidata_recommender
    $ virtualenv myenvironment
    $ source myenvironment/bin/activate
    $ python install -r requirements.txt
    
You also have to install a Celery worker server to process Celery tasks. We use [RabbitMQ](http://docs.celeryproject.org/en/latest/getting-started/brokers/rabbitmq.html#broker-rabbitmq).

We periodically execute the tasks `refresh_data`, which reads up-to-date necessary information from the database, and `train`: trains the model with given hyperparameters. The recommendations are also produced by a Celery task called `recommend`.
    


## Contributing

The Afinidata Content Manager is a Free Software Product created by Afinidata and available under the AGPL Licence. 

To contribute, read our [Code of Conduct](CODE_OF_CONDUCT.md), our [Docs at Read The Docs](https://afinidata-content-manager.readthedocs.io/en/latest/) and code away.
Create a pull request and contact us in order to merge your suggested changes. We suggest the use of git flow in order to provide a better contributing experience.






