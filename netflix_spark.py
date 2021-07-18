#=== Configuration steps
import pandas as pd
import numpy as np
import os
# Path for spark source folder
os.environ['SPARK_HOME']="/opt/spark"
os.environ['SPARK_LOCAL_IP']="spark://kelvinhwee-VirtualBox:7077"

netflix_path = r'/media/sf_Z._Shared_folder_for_Ubuntu'
os.chdir(netflix_path)

from functools import reduce

import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row, DataFrame

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col, lit, when
from pyspark.sql.functions import isnan, when, count, col, array
from pyspark.sql.functions import date_format, to_date
from pyspark.sql.functions import array_contains, array_sort

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

try: sc.stop(); print("spark context stopped")
except: pass

try: spark.stop(); print("spark stopped")
except: pass


#=== create the Spark config, Spark session, sparkContext, and sqlContext
#- Spark Config
master = "spark://kelvinhwee-VirtualBox:7077"
config = SparkConf().setAll([('spark.executor.memory', '1g'), # controls "memory per executor" shown in SparkUI; we configured "1g" per worker
                             ('spark.executor.cores', '1'), # affects "Cores" in SparkUI; we configured 1 core for the worker
                             ('spark.cores.max', '2'),
                             ('spark.driver.memory', '2g')]) \
                    .setMaster(master) \
                    .set("spark.driver.port", "10027") \
                    .setAppName("Netflix_recsys")


#- Spark Session
spark = SparkSession.builder.config(conf = config).getOrCreate()

#- Apache Arrow settings
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

#- Spark Context
sc = spark.sparkContext
sc._conf.getAll() # print the configurations being set
print(sc.defaultParallelism) # print the number of cores (1)

#- SQL Context
sqlContext = SQLContext(spark.sparkContext)


#=== create RDD from the lines read into the text file

#--- read in the text file as an RDD
rdd = sc.textFile('data.txt')  #rdd = sc.textFile('data.txt', 2000)
rdd.take(5)

#--- split the row into separate items
rdd2 = rdd.map(lambda line : line.split(","))
rdd2.take(5)

#=== convert RDD into Spark dataframe (movie ratings dataframe)
netflix_spark_df = sqlContext.createDataFrame(rdd2)
netflix_spark_df.printSchema()

cols_to_drop = ("_1", "_2", "_3", "_4")

netflix_spark_df2 = netflix_spark_df.withColumn("Movie_ID", netflix_spark_df._1.cast(IntegerType())) \
                                    .withColumn("Customer_ID", netflix_spark_df._2.cast(IntegerType())) \
                                    .withColumn("Ratings", netflix_spark_df._3.cast(IntegerType())) \
                                    .withColumn("Date_of_review", netflix_spark_df._4.cast(DateType())) \
                                    .drop(*cols_to_drop) # putting "*" converts everything into positional args

netflix_spark_df2.show(5)


#=== read the movie titles CSV file (for Netflix and IMDB)
#--- define the schema (for Netflix movies)
schema_netflix = StructType([
                            StructField("Movie_ID", IntegerType(), nullable=False),
                            StructField("Movie_year", IntegerType(), nullable=False),
                            StructField("Movie_title", StringType(), nullable=False)
                            ])

movie_titles_spark_df = spark.read.option("header", False) \
                             .csv(path = netflix_path + '/movie_titles.csv',
                                  schema = schema_netflix).cache()


#--- define the schema (for IMDB movies)
schema_imdb = StructType([
                         StructField("IMDB_title_id", StringType(), nullable=False),
                         StructField("Title", StringType(), nullable=False),
                         StructField("Original_title", StringType(), nullable=False),
                         StructField("Year", IntegerType(), nullable=False),
                         StructField("Date_published", DateType(), nullable=False),
                         StructField("Genre", StringType(), nullable=False),
                         StructField("Duration", IntegerType(), nullable=False)
                         ])

IMDB_spark_df = spark.read.option("header", False) \
                             .csv(path = netflix_path + '/IMDb movies.csv',
                                  schema = schema_imdb).cache()


#=== join the movie ratings dataframe with the movie titles dataframe
joined_df = netflix_spark_df2.join(movie_titles_spark_df, ["Movie_ID"]) # putting the join condition in list drops duplicates
joined_df2 = joined_df.join(IMDB_spark_df, movie_titles_spark_df["Movie_title"] == IMDB_spark_df["Title"])
joined_df3 = joined_df2.select(["Movie_ID","Movie_title","Movie_year",
                                "Customer_ID","Ratings","Date_of_review",
                                "Genre","Duration"])

#=== convert Spark dataframe to RDD
joined_rdd = joined_df3.rdd.map(list)

#=== create the rows for the matrix setup (to determine user similarity)
#--- get a list of movie IDs
distinct_movie_id_rdd = joined_df3.select("Movie_ID").distinct().rdd
distinct_movie_id_list = distinct_movie_id_rdd.map(lambda r : r[0]).collect()
distinct_movie_id_list2 = sorted(distinct_movie_id_list)
len_movie_id = len(distinct_movie_id_list2)


#--- get a list of customer IDs
distinct_cust_id_rdd = joined_df3.select("Customer_ID").distinct().rdd
distinct_cust_id_list = distinct_cust_id_rdd.map(lambda r : r[0]).collect()
distinct_cust_id_list2 = sorted(distinct_cust_id_list)
len_cust_id = len(distinct_cust_id_list2)


### TESTING AREA




par_x = sc.parallelize([1,2,3,4,5])
type(par_x)
par_x.collect()


df = spark.createDataFrame(
    sc.parallelize([["id" + str(n)] + np.random.randint(0, 2, 10).tolist() for n in range(20)]),
    ["id"] + ["movie" + str(i) for i in range(10)])
df.show()


### TESTING AREA