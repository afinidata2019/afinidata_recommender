from collections import OrderedDict
from datetime import datetime
import pickle

import pandas as pd
from pandas.api.types import CategoricalDtype


class PreprocessGenericData(object):
    def __init__(self, pipeline):
        """
        Defines a list of preprocessing tasks to be performed on a Pandas dataframe with a
        temporal nature. The rows in the dataframe correspond to `events` classified according to
        different `types`.

        :param pipeline: OrderedDict of the form:
            task: argument
        where task must be a method in the class and argument must be the method's arguments.
        """
        for task in pipeline:
            assert hasattr(self, task), '{} is not a method in the class.'.format(task)
        self.pipeline = OrderedDict(pipeline)

    _basic_dtypes = ['int64', 'float64', 'datetime64', 'category']

    @classmethod
    def set_dtypes(cls, df, dtype_specs):
        """
        Transform dataframe columns into their corresponding types. This is necessary,
        for example, when a CSV file is imported and datetime columns are in a string format.

        :param df: Pandas dataframe
        :param dtype_specs: list of tuples with entries
            (column, dtype)
         where column is a string specifying the name of a dataframe column and  dtype is another
         string specifying the expected type the elements in the column should have. Supported
         dtype strings are drawn from Pandas basic dtypes:
         # https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dtypes

        :return: transformed dataframe.
        """

        for (column, dtype) in dtype_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)
            assert dtype in cls._basic_dtypes, '{} is not a (basic) supported type for conversion'.format(dtype)

            try:
                df[column] = df[column].astype(dtype, errors='ignore')
            except ValueError as ve:
                print(ve)

        return df

    @staticmethod
    def drop_rows_with(df, value_specs):
        """
        Drop all rows where a given column has a particular value.

        :param df: Pandas dataframe
        :param value_specs: list of tuples with entries of the form
            (column, [value1, value2, ...])
         where column is the name of the dataframe column and values in the list are those to be
         discarded.
        :return: reduced dataframe
        """

        for column, values in value_specs:
            assert column in df.columns, '{} is not a column in the dataframe'.format(column)

            for value in values:
                df = df.drop(df[df[column] == value].index)

        return df

    def execute_pipeline(self, df):
        """

        :param df: pandas dataframe on which to execute pipeline.
        :return: pandas dataframe resulting from preprocessing.
        """
        for task in self.pipeline:
            args = self.pipeline[task]
            if args is not None:
                df = getattr(self, task)(df, args)
            else:
                df = getattr(self, task)(df)

        return df


class PreprocessInteractionData(PreprocessGenericData):
    def __init__(self, pipeline):
        super().__init__(pipeline)


class SetUpDataframes(object):
    response_pipeline = {
        'set_dtypes': [('response', 'int64')],
        'drop_rows_with': [('response', [0])]
    }

    @classmethod
    def response_df(cls, raw_df):
        """
        Setup the responses dataframe.

        :param raw_df: Raw dataframe directly read from the MySQL database by the feedback_data
         method in the ReadDatabase class.
        :return: feedback pandas dataframe ready for use.
        """
        preprocessor = PreprocessInteractionData(cls.response_pipeline)
        return raw_df.pipe(preprocessor.execute_pipeline)

    @classmethod
    def response_matrix(cls, raw_df):
        """
        Produce feedback matrix with rows associated to posts, columns associated to users and entries
        given by the feedback given by a user to a post.

        :param raw_df: raw feedback dataframe.
        :return: pandas dataframe.
        """
        return cls.response_df(raw_df).pivot_table(
            index='question_id', columns='user_id', values='response', aggfunc='mean')
