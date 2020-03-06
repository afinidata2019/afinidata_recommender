from afinidata_recommender.app import init_celery
from celery.schedules import crontab

app = init_celery()
app.conf.imports = app.conf.imports + ('afinidata_recommender.tasks',)
