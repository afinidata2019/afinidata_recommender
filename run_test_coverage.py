import coverage
from afinidata_recommender.tasks import tasks

cov = coverage.coverage()
cov.erase()
cov.start()


tasks.train(epochs=200)
tasks.refresh_data()
tasks.recommend(5, 10)
tasks.recommend(-1, 0)


cov.stop()
cov.save()
cov.report()