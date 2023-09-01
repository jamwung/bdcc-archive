import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField

from pyspark.sql import SparkSession

spark = (
    SparkSession
    .builder
    .master('local[*]')
    .getOrCreate()
)

# trips = spark.read.csv(
#     '/mnt/data/public/nyctaxi/all/green_tripdata_2017-1*.csv',
#     sep=',', header=True, inferSchema=True
# )

# tweets = spark.read.json(
#     '/mnt/data/public/twitter/sample/data-1909302*.json.bz2'
# )

def get_a_counter_udf():
    """
    Return a udf that accepts a string column and returns the number of times,
    asn an integer, that the case-insensitive letter a is found in the column.
    """
    @udf('int')
    def counter(string):
        """
        Count the number of occurences of case-insensitive a in the string.
        """
        return string.lower().count('a') if string is not None else 0
    return counter

def get_zscore_udf():
    """
    Return a pandas udf.
    """
    @pandas_udf('double')
    def zscore(s: pd.Series) -> pd.Series:
        """
        Accept a float column and return the z-score.
        """
        return (s - s.mean()) / s.std()
    return zscore

def get_fare_udf():
    """
    Return a pandas udf.
    """
    schema = """
        STRUCT <
            fare_amount: FLOAT,
            extra: FLOAT,
            mta_tax: FLOAT,
            tip_amount: FLOAT,
            tolls_amount: FLOAT,
            ehail_fee: FLOAT,
            improvement_surcharge: FLOAT,
            total_amount: FLOAT
        >
    """
#     schema = StructType([
#         StructField('fare_amount', FloatType()),
#         StructField('extra', FloatType()),
#         StructField('mta_tax', FloatType()),
#         StructField('tip_amount', FloatType()),
#         StructField('tolls_amount', FloatType()),
#         StructField('ehail_fee', FloatType()),
#         StructField('improvement_surcharge', FloatType()),
#         StructField('total_amount', FloatType())
#     ])
    @pandas_udf(returnType=schema)
    def get_struct(fare_amount: pd.Series, extra: pd.Series,
                   mta_tax: pd.Series, tip_amount: pd.Series,
                   tolls_amount: pd.Series, ehail_fee: pd.Series,
                   improvement_surcharge: pd.Series, total_amount: pd.Series)\
    -> pd.DataFrame:
        """
        Return a struct with float fields corresponding to the inputs.
        """
        #try StructType.fieldnames()
        return pd.DataFrame({
            'fare_amount': fare_amount,
            'extra': extra,
            'mta_tax': mta_tax,
            'tip_amount': tip_amount,
            'tolls_amount': tolls_amount,
            'ehail_fee': ehail_fee,
            'improvement_surcharge': improvement_surcharge,
            'total_amount': total_amount
        })
    return get_struct

def ave_amount():
    """
    Return an SQL statement that will process trips to create a table with
    columns PULocationID and mean fare amount, which contains the average
    fare amount per PULocationID. Sort the table by PULocationID
    """
    return """
        SELECT
            PULocationID,
            AVG(fare_amount) as `mean fare amount`
        FROM trips
        GROUP BY PULocationID
        ORDER BY PULocationID
    """

def pickup():
    """
    Return an SQL statement to process trips into a DataFrame with columns
    pickup_hour and pickup_location. Sort the resulting dataframe by
    pickup_hour.
    
    Columns
    -------
    pickup_hour : string
        corresponds to the lpep_pickup_datetime rounded down to the nearest
        hour
    pickup_locations : array
        array of unique PULocationID for that pickup_hour sorted by
        PULocationID
    """
#     return """
#         SELECT
#             CAST(DATE_TRUNC('hour', lpep_pickup_datetime) as STRING)
#         FROM trips
#         LIMIT 10
#     """
    return """
        SELECT
            CAST(
                date_trunc('hour', lpep_pickup_datetime) AS STRING
            ) as pickup_hour,
            sort_array(collect_set(PULocationID), TRUE) as pickup_locations
        FROM
            trips
        GROUP BY
            pickup_hour
        ORDER BY
            pickup_hour
    """

def friendliest():
    """
    Return an SQL statement to process tweets into a DataFrame with columns
    screen_name and friends_count and rows corresponding to the 10 unique
    screen_names with the most friends_count. Sort by descending friends_count
    then by screen_name.
    """
    return """
        SELECT
            DISTINCT user.screen_name,
            user.friends_count
        FROM tweets
        ORDER BY
            friends_count DESC,
            screen_name
        LIMIT 10
    """

def hashiest():
    """
    Return an SQL statement to process tweets into a Spark DataFrame with
    columns hashtag and count, and rows corresponding to the 10 most common
    hashtags sorted by descending frequency then by hashtag.
    """
    return """
        SELECT
            hashtag,
            COUNT(*) AS count
        FROM (
            SELECT
                explode(entities.hashtags.text) AS hashtag
            FROM
                tweets
        )
        GROUP BY
            hashtag
        ORDER BY
            count DESC,
            hashtag
        LIMIT 10
    """

def count_tokens():
    """
    Return an SQL statement to process a Spark DataFrame of filepaths
    and text into a dataframe with columns id and tokens sorted by id.
    
    Columns
    -------
    id : string
        corresponds to the id of the book
    tokens : int
        corresponds to the number of tokens in each book, including headers
        and footers. A token is defined as a sequence of non-whitespace
        characters.
    """
    return """
        SELECT
            regexp_extract(
                CAST(input_file_name() AS STRING), '([0-9]{5}).txt$', 1
            ) as id,
            size(
                array_remove(split(value, r'\s+'), '')
            ) as tokens
        FROM texts
        ORDER BY
            id
    """

def count_love():
    """
    Return a SQL statement to process a Spark DataFrame of strings into a
    DataFrame with columns has love and count.
    
    Columns
    -------
    `has love` : string
        should have row values loveful and loveless, in that order.
    `count` : int
        the number of rows that has the word love (case-insensitive, bounded
        by non-word characters) and the words that do not have it,
        respectively.
    """
    return """
        SELECT
            CASE
                WHEN
                    value REGEXP r'(?i)\\blove\\b'
                THEN
                    'loveful'
                ELSE
                    'loveless'
            END as `has love`,
            COUNT(*) as count
        FROM texts
        GROUP BY
            `has love`
    """

def count_creation():
    """
    Return an SQL statement to process tweets2 into a Spark DataFrame with
    columns year corresponding to the creation year of users and frequency
    corresponding to the number of unique users with that creation year. Sort
    them by descending number of users.
    """
    return '''
        SELECT
            CAST(SUBSTRING(user.created_at, -4, 4) AS INT) AS year,
            COUNT(DISTINCT user.id) AS frequency
        FROM tweets2
        WHERE created_at IS NOT NULL
        GROUP BY
            year
        ORDER BY
            frequency DESC
    '''