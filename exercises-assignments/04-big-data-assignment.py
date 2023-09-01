import time
import re
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import udf

import warnings
warnings.filterwarnings("ignore")
import pyspark.pandas as ps
from pyspark.sql import SparkSession

spark = (
    SparkSession
    .builder
    .master('local[*]')
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
    .getOrCreate()
    )
spark.sparkContext.setLogLevel('OFF')

def mean_daily_views():
    """
    Return a Spark DataFrame sorted by day of month.
    
    Columns
    -------
    day : int
        corresponds to day of month
    mean daily views : float
        corresponds to the mean page views per day of the month for the
        English Wikipedia (en.z) article Big_Data (exact match)
    """
    schema = '''
        project STRING,
        page STRING,
        monthly_total INT,
        hourly_counts STRING
    '''
    fp = ('/mnt/localdata/public/wikipedia/pageviews/'
          'pagecounts-2020-??-views-ge-5.bz2')
    return (
        SparkSession.getActiveSession()
        .read.option('sep', ' ').schema(schema).csv(fp)
        .filter((F.col('project') == 'en.z') & (F.col('page') == 'Big_Data'))
        .withColumn('daily_counts',
                    col=F.explode(F.split(F.col('hourly_counts'), ',')))
        .filter(F.col('daily_counts') != '')
         .select(
             F.aggregate(
                 F.split(
                     F.substring('daily_counts', 3, 100_000),
                     r'\D'
                 ).cast('array<int>'),
                 F.lit(0),
                 lambda acc, x: acc + x
             ).alias('count'),
             (F.ascii('hourly_counts') - ord('A') + 1).alias('day')
         )
         .groupby('day')
         .agg(F.avg('count').alias('mean daily views'))
         .orderBy('day')
    )

def mean_length_per_year():
    """
    Return a SQL statement to process the registered Spark DataFrame with
    columns year and mean_length. The column year corresponds to the year
    column and the column mean_length  Sort by year.
    
    Columns
    -------
    year : int
        corresponds to the year column
    mean_length : float
        corresponds to the n-gram length (in characters) for that year. Only
        n-grams made up of all letters after the underscore and part-of-speech
        have been removed are considered.
    """
    # alternative: WHERE ngram REGEXP '^[A-z]+_[A-Z]*$'
    return """
    SELECT 
        year,
        AVG(LENGTH(ngram_new)) AS mean_length
    FROM (
        SELECT
            year,
            REGEXP_EXTRACT(ngram, '([^_]+)_?') AS ngram_new
        FROM
            ngrams
        WHERE NOT
            REGEXP(REGEXP_EXTRACT(ngram, '([^_]+)_?', 1), '[^a-zA-Z]+')
    )
    GROUP BY
        year
    ORDER BY
        year
    """


def comments_by_account_creation(df):
    """
    Return a pandas-on-Spark Series with index corresponding to the year-month
    (in UTC) of the account creation of the user who posted the comment, and
    value corresponding to the number of comments authored by users created
    that year-month. Sort in reverse chronological order of account created
    and return only the 12 most recent months.
    """
    return (
        df
        .assign(
            account_created=(ps.to_datetime(df.author_created_utc, unit='s')
                             .dt.strftime('%Y-%m'))
        )
        .groupby('account_created')
        ['subreddit_id']
        .count()
        .sort_index(ascending=False)
        .head(12)
    )