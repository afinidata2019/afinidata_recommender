import logging
import os

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine

from recommender.read_db import ReadDatabase
from recommender.preprocess import SetUpDataframes
from recommender.models import CollaborativeFiltering
from recommender.datasets import Datasets


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

load_dotenv('.env')

# set up database reader
DB_URI = os.environ.get("DB_URI")
engine = create_engine(DB_URI)

reader_cm = ReadDatabase(engine, 'CM_BD')

question_df = reader_cm.get_data('id, post_id', 'posts_question', None).set_index('id')

taxonomy_df = reader_cm.get_data('post_id, area_id', 'posts_taxonomy', None)
taxonomy_areas = taxonomy_df.groupby('area_id')

content_df = reader_cm.get_data('id, min_range, max_range', 'posts_post', None)

interaction_df = reader_cm.get_data('user_id, post_id', 'posts_interaction', "type IN ('sended', 'sent')")
interaction_df = interaction_df[~interaction_df['post_id'].isna()]
interaction_df['post_id'] = interaction_df['post_id'].astype('int32')




app = Celery('tasks', broker='pyamqp://guest@localhost//')

# We would like to enable the following sequence of background tasks:
#


@app.task
def train(epochs=10000, lr=0.00001, alpha=0., depth=1):
    # extract data from posts_response into a pandas dataframe and
    # slightly process only relevant data for training
    # in this case, so far we are only considering data for which
    # there is an alpha value in the 'response' column
    try:
        response_df = reader_cm.get_data(
            'user_id, response, question_id', 'posts_response',
            "created_at >= '2019-09-20'",
            None)
    except Exception as e:
        logger.exception("Unable to retrieve data from the database. Check your internet connection " +
                         "to the database or the parameters in the SQL query. Error: %s", e)

    response_df = response_df[
        (response_df['response'].apply(lambda x: x.isdigit())) & (response_df['response'] != '0')]
    response_df = response_df.drop_duplicates().reset_index(drop=True)
    print('*' * 80)
    print(f'total number of responses in response_df: {len(response_df)}')

    # create matrix for training with items over rows and users over columns
    # as a numpy matrix
    response_matrix = SetUpDataframes.response_matrix(response_df)

    # train test split
    datasets = Datasets(response_matrix)
    train_set, test_set = datasets.train_test_split(0.10)

    # model initialization
    model = CollaborativeFiltering()
    model.actors = {
        'users': response_matrix.columns.values,
        'items': response_matrix.index.values
    }
    model.n_items = len(datasets.posts)
    model.n_users = len(datasets.users)

    model.train(
        train_matrix=train_set,
        test_matrix=test_set,
        epochs=epochs,
        alpha=alpha,
        n_features=depth,
        lr=lr,
        resume=False
    )

    print('*' * 80)
    model.save_model(f'afinidata_recommender_model_specs')
    print(f'model has been saved to afinidata_recommender_model_specs.pkl in the local directory')


@app.task
def user_to_activities_df():
    reader_ms = ReadDatabase(engine, 'contentManagerUsers')
    child_dob = reader_ms.get_data('parent_user_id, dob', 'messenger_users_child')


@app.task
def recommend(user_id, months):
    # model initialization and load
    model = CollaborativeFiltering()

    model.load_model('afinidata_recommender_model_specs')
    ranking = model.afinidata_recommend(user_id=user_id, question_df=question_df, taxonomy_df=taxonomy_df)

    content_for_age = content_df[(content_df['min_range'] <= months) & (content_df['max_range'] >= months)][
        'id'].values.tolist()
    sent_activities = interaction_df[interaction_df['user_id'] == user_id]['post_id'].values.tolist()
    return ranking[(ranking['post_id'].isin(content_for_age)) & (~ranking['post_id'].isin(sent_activities))]