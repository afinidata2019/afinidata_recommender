import logging
import os
import pickle

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine

from afinidata_recommender.recommender.read_db import ReadDatabase
from afinidata_recommender.recommender.models import CollaborativeFiltering
from afinidata_recommender.recommender.preprocess import SetUpDataframes
from afinidata_recommender.recommender.datasets import Datasets


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

load_dotenv('.env')

# environment variables
DB_URI = os.environ.get("DB_URI")
CELERY_BROKER = os.environ.get('CELERY_BROKER', 'pyamqp://guest@localhost/')

# set up database reader
engine = create_engine(DB_URI)
reader_cm = ReadDatabase(engine, 'CM_BD')

app = Celery('tasks', backend="rpc", broker=CELERY_BROKER)


@app.task
def refresh_data():
    """
    Task that reloads particular neccesary data from the db and saves it to pickle files.
    """
    question_df = reader_cm.get_data(
        'id, post_id',
        'posts_question',
        None)

    taxonomy_df = reader_cm.get_data(
        'post_id, area_id',
        'posts_taxonomy',
        None)

    content_df = reader_cm.get_data(
        'id, min_range, max_range',
        'posts_post',
        "status IN ('published')")

    response_df = reader_cm.get_data(
        'user_id, response, question_id',
        'posts_response',
        "created_at >= '2019-09-20'",
        None)
    response_df = response_df[
        (response_df['response'].apply(lambda x: x.isdigit())) & (response_df['response'] != '0')]
    response_df = response_df.drop_duplicates().reset_index(drop=True)

    pickle.dump(question_df, open("question.pkl", "wb"))
    pickle.dump(taxonomy_df, open("taxonomy.pkl", "wb"))
    pickle.dump(content_df, open("content.pkl", "wb"))
    pickle.dump(response_df, open("response.pkl", "wb"))


@app.task
def train(epochs=10000, lr=0.00001, alpha=0., depth=2):
    """
    Train the collaborative filtering model according to the specified parameters and saves it to a
    pickle file.
    """
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
    logger.info('*' * 80)
    logger.warning(f'total number of responses in response_df: {len(response_df)}')

    # create matrix for training with items over rows and users over columns
    # as a numpy matrix
    response_matrix = SetUpDataframes.response_matrix(response_df)

    # train test split
    datasets = Datasets(response_matrix)
    train_set, test_set = datasets.train_test_split(0.0)

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

    logging.info('*' * 80)
    model.save_model(f'afinidata_recommender_model_specs')
    logging.warning(f'model has been saved to afinidata_recommender_model_specs.pkl in the local directory')


@app.task
def recommend(user_id, months):
    model = CollaborativeFiltering()

    model.load_model('afinidata_recommender_model_specs')
    question_df, taxonomy_df, content_df, response_df =\
    (pickle.load(open(file_name, "rb")) for file_name in
     ["question.pkl", "taxonomy.pkl", "content.pkl", "response.pkl"])

    sent_activities = reader_cm.get_data(
        'post_id',
        'posts_interaction',
        f"type IN ('sended', 'sent', 'dispatched') AND user_id={user_id}")['post_id'].unique().tolist()

    ranking = model.afinidata_recommend(
        user_id=user_id,
        months=months,
        question_df=question_df,
        taxonomy_df=taxonomy_df,
        content_df=content_df,
        response_df=response_df,
        sent_activities=sent_activities)

    return ranking.to_json()