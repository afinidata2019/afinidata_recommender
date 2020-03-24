#!/usr/bin/env bash

## A script to launch workers for recommender on recommender task queue
celery -A afinidata_recommender.tasks.tasks worker --concurrency=8  --loglevel=info
