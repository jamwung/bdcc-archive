from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import RandomForestRegressor, RandomForestRegressorModel

from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.functions import collect_set
from pyspark.ml.fpm import FPGrowth

from pyspark.ml.recommendation import ALS

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import parallel_backend
from joblibspark import register_spark
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

spark = (
    SparkSession
    .builder
    .master('local[*]')
    .getOrCreate()
)

def corr(df, col1, col2):
    """
    Return the Spearman correlation of two Spark DataFrame columns.
    """
    assembler = VectorAssembler(inputCols=[col1, col2], outputCol="features")
    df_corr = Correlation.corr(
        dataset=assembler.transform(df),
        column='features',
        method='spearman'
    )
    return df_corr.first()[0][0, 1]

def predict_payment_type(df_training, df_to_predict):
    """
    Accepts the training and prediction data in the form of Spark DataFrames
    then returns the prediction as a list.
    """
    df_training = (
        df_training
        .withColumn('PULocationID', df_training.PULocationID.astype('double'))
        .withColumn('DOLocationID', df_training.DOLocationID.astype('double'))
        .withColumn('payment_type', df_training.payment_type.astype('int'))
    )
    
    assembler = VectorAssembler(inputCols=['PULocationID', 'DOLocationID'],
                                outputCol='features')

    df_training = assembler.transform(df_training)

    vi = VectorIndexer(maxCategories=10_000,
                       inputCol='features',
                       outputCol='features_indexed') 
    vi_trained = vi.fit(df_training)
    
    df_training = vi_trained.transform(df_training)
    
    df_to_predict = assembler.transform(df_to_predict)
    df_to_predict = vi_trained.transform(df_to_predict)

    rf = RandomForestClassifier(
        featuresCol='features_indexed',
        labelCol='payment_type',
        maxBins=1000,
        seed=2020
    )
    rf_trained = rf.fit(df_training)
    
    df_predicted = rf_trained.transform(df_to_predict)
    return (
        df_predicted
        .select(df_predicted.prediction.astype('int'))
        .toPandas()
        .squeeze()
        .tolist()
    )