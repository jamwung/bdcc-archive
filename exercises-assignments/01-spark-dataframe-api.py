import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.functions import explode, desc

from pyspark.sql.types import (IntegerType, TimestampType, FloatType,
                               StringType, StructType, StructField)

spark = (
    SparkSession.builder
    .master('local[*]')
    .getOrCreate()
)

# trips = spark.read.csv(
#     '/mnt/data/public/nyctaxi/all/green_tripdata_2017-1*.csv',
#     sep=',', header=True
# )

# tweets = spark.read.json(
#     '/mnt/data/public/twitter/sample/data-1909302*.json.bz2'
# )

def pudo(trips):
    """
    Return a triple corresponding to the PULocationID, DOLocationID, and count
    of the most frequent PULocationID-DULocationID pair.
    """
    return tuple(
        trips
        .groupBy('PULocationID', 'DOLocationID')
        .count()
        .orderBy(F.desc('count'))
        .first()
    )

def ave_amount(df_trips):
    """
    Return a Spark DataFrame with columns PULocationID and mean fare amount
    which contains the average fare amount per PULocationID. Sort by
    PULOcationID.
    """
    return (
        trips
        .groupby('PULocationID')
        .agg(F.mean('fare_amount').alias('mean fare amount')) # alternative
#         .mean('fare_amount')
#         .withColumnRenamed('avg(fare_amount)', 'mean fare amount')
        .orderBy('PULocationID')
    )

def pickup(trips):
    """
    Return a Spark DataFrame with columns pickup_hour and pickup_location.
    Sort the dataframe by pickup_hour.
    
    Columns
    -------
    pickup_hour : string
        corresponds to lpep_pickup_datetime rounded down the the nearest hour
    pickup_location : array
        array of unique PULocationID for that pickup_hour sorted by
        PULocationID.
    """
    return (
        trips
        .withColumn('pickup_hour',
                    F.date_trunc('hour', trips.lpep_pickup_datetime)
                    .cast('string'))
        .groupBy('pickup_hour')
        .agg(F.sort_array(F.collect_set('PULocationID'))
             .alias('pickup_locations'))
        .orderBy('pickup_hour')
    )

def friendliest(df_tweets):
    """
    Return a Spark DataFrame with columns screen_name and friends_count with
    rows corresponding to the ten unique screen_names with the most
    friends_count sorted by decreasing friends_Count then screen_name.
    """
    return (
        df_tweets
        .select('user.screen_name', 'user.friends_count')
        .distinct()
        .orderBy(F.desc('user.friends_count'), 'user.screen_name')
        .limit(10)
    )

def hashiest(df_tweets):
    """
    Return a Spark DataFrame with columns hashtag and count, and rows
    corresponding to the ten most common hashtags sorted by decreasing
    frequency then by hashtag.
    """
    return (
        df_tweets
        .select(F.explode('entities.hashtags.text').alias('hashtag'))
        .groupby('hashtag')
        .agg(F.count('hashtag').alias('count'))
        .orderBy(['count', 'hashtag'], ascending=[False, True])
        .limit(10)
    )

def geek_tweets(df_tweets):
    """
    Accept a Spark DataFrame and save the tweets that have the
    case-insensitive word geek in their user description to a parquet file
    directory geek-tweets partitioned by created_at.  
    """
    return (
        df_tweets
        .filter(F.col('user.description').rlike(r'(?i)\bgeek\b'))
        .write.partitionBy('created_at').parquet('geek-tweets',
                                                 mode='overwrite')
    )

def count_tokens(fpaths):
    """
    Return a Spark DataFrame with columns id and tokens sorted by id.
    
    Columns
    -------
    id : int
        id of the book
    tokens : string
        corresponds to the number of tokens in each book, including headers
        and footers. A token is a sequence of non-whitespace charactes.
    """
    spark = SparkSession.builder.getOrCreate()
    return (
        spark.read.text(fpaths, wholetext=True)
        .withColumn('filename', F.input_file_name())
        .select(
            F.regexp_extract('filename', r'(\d{5})\.txt$', 1).alias('id'),
            F.size(
                F.array_remove(F.split('value', '\s+'), '')
            ).alias('tokens')
        )
        .orderBy('id')
    )

def count_love(texts):
    """
    Return the number of occurences of the case-insensitive word love bounded
    by non-word characters in all of the strings.
    """
    return (
        texts
        .select(F.explode(F.split('value', '\n')).alias('tokens'))
        .filter(F.col('tokens').rlike(r'(?i)\blove\b'))
        .count()
    )