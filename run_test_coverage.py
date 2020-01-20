
import coverage
from afinidata_recommender.tasks import tasks

cov = coverage.coverage()
cov.erase()
cov.start()


tasks.train()
tasks.refresh_data()
tasks.recommend(5, 10)


cov.stop()
cov.save()
cov.report()