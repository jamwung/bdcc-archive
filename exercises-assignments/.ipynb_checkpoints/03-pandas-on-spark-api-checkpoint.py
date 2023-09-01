import pyspark.pandas as ps

from pyspark.sql import SparkSession

spark = (
    SparkSession
    .builder
    .master('local[*]')
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
    .getOrCreate()
)


df_trips = spark.read.csv(
    '/mnt/data/public/nyctaxi/all/green_tripdata_2017-1*.csv',
    header=True
)

def pudo(df_trips):
    """
    Return a triple corresponding to PULocationID, DOLocationID, and count of
    the most frequent PULocationID - DOLocationID pair using only a single
    pandas-on-spark DataFrame method call chain.
    """
    return tuple(
        df_trips
        .groupby(['PULocationID', 'DOLocationID'])
        .size()
        .nlargest(1)
        .reset_index()
        .to_numpy().flatten()
    )

def ave_amount(df_trips):
    """
    Return a pandas-on-spark DataFrame with columns PULocationID and
    mean fare amount which contains the average fare amount per PULocationID
    sorted by PULocationID.
    """
    return (
        df_trips
        .groupby('PULocationID')['fare_amount']
        .mean()
        .sort_index()
        .reset_index()
        .rename({'fare_amount': 'mean fare amount'}, axis=1)
    )

def pickup(df_trips):
    """
    Return a pandas-on-spark DataFrame with columns pickup_hour and
    pickup_location sorted by pickup_hour.
    
    Columns
    -------
    pickup_hour : string
        corresponds to lpep_pickup_datetime rounded down the the nearest hour
    pickup_location : array
        array of unique PULocationID for that pickup_hour sorted by
        PULocationID.
    """
    return (
        df_trips
        .assign(
            pickup_hour=(df_trips.lpep_pickup_datetime.dt.floor('1H')
                         .astype(str))
        )
        .groupby('pickup_hour')
        ['PULocationID']
        .apply(lambda x: sorted(set(x)))
        .sort_index()
    )

def priciest(df_trips):
    """
    Return a pandas-on-spark DataFrame with columns PULocationID and
    DOLocationID and rows corresponding to the 10 unique
    PULocationID-DOLocationID with the highest fare_amount sorted by
    descending fare_amount.
    """
    return (
        df_trips
        .groupby(['PULocationID', 'DOLocationID'])
        ['fare_amount']
        .max()
        .sort_index()
        .nlargest(10)
        .reset_index()
        [['PULocationID', 'DOLocationID']]
    )

def pickup_hour_minute(df_trips):
    """
    Return a pandas-on-spark DataFrame only for rows with lpep_pickup_datetime
    date 30 December 2017 with new columns pickup_hour and pickup_minute
    corresponding to the lpep_pickup_datetime hour and minute, respectively,
    sorted by lpep_pickup_datetime.
    """
    return (
        df_trips[df_trips.lpep_pickup_datetime.dt.date == '2017-12-30']
        .assign(
            **{'pickup_hour': df_trips['lpep_pickup_datetime'].dt.hour,
               'pickup_minute': df_trips['lpep_pickup_datetime'].dt.minute}
        )
        .sort_values('lpep_pickup_datetime')
    )

def pickup_unique_per_hour(df_trips):
    """
    Return a pandas-on-Spark DataFrame with index corresponding to the hour of
    day of lpep_pickup_datetime and columns PULocationID and DOLocationID,
    corresponding to the number of unique values for those columns at that
    hour of day. Sort by lpep_pickup_datetime hour.
    """
    return (
        df_trips
        .assign(**{'hour of day': df_trips.lpep_pickup_datetime.dt.hour})
        .groupby('hour of day')
        [['PULocationID', 'DOLocationID']]
        .agg('nunique')
        .sort_index()
    )

def pudo_unique_per_hour(df_trips):
    """
    Return a pandas-on-spark Series with index corresponding to
    lpep_pickup_datetime hour and values corresponding to the number of unique
    PULocationID-DULocationID pairs for that hour sorted by
    lpep_pickup_datetime hour.
    """
    return (
        df_trips
        .assign(
            hour=df_trips.lpep_pickup_datetime.dt.hour,
            pudo_pair=(df_trips.PULocationID.astype(str) + '-' +
                       df_trips.DOLocationID.astype(str))
        )
        .groupby('hour')['pudo_pair']
        .nunique()
        .sort_index()
    )

def pair_pudo():
    """
    Returns a function that accepts a dataframe and returns a dataframe having
    only one column named pudo that corresponds to the PULocationID and
    DOLocationID concatenated by a hyphen.
    """
    def apply_func(df) -> ps.DataFrame['pudo': str]:
        series = (df['PULocationID'].astype(str) + '-' +
                  df['DOLocationID'].astype(str))
        return series.to_frame()
    return apply_func

def plot_trips_per_day(df_trips):
    """
    Return a pyplot Figure of a bar plot of the number of trips vs day of week
    according to lpep_pickup_datetime.
    """
    return (
        df_trips
        .assign(**{'day of week': df_trips.lpep_pickup_datetime.dt.dayofweek})
        .groupby('day of week')
        .size()
        .sort_index()
        .rename('number of trips')
        .reset_index()
        .plot.bar(x='day of week', y='number of trips')
    )

def mean_tip(df_trips):
    """
    Return a pandas-on-Spark DataFrame with index corresponding to the
    lpep_pickup_datetime hour, columns corresponding to the
    lpep_dropoff_datetime and values corresponding to the mean tip_amount or
    np.nan if that combination was not found. Sort indices and columns.
    """
    return (
        df_trips
        .assign(pu_hour=df_trips.lpep_pickup_datetime.dt.hour,
                do_hour=df_trips.lpep_dropoff_datetime.dt.hour)
        .groupby(['pu_hour', 'do_hour'])
        ['tip_amount']
        .mean()
        .reset_index()
        .pivot(index='pu_hour', columns='do_hour', values='tip_amount')
        .sort_index()
    )