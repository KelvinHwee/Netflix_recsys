#=== Configuration steps
import pandas as pd
import numpy as np
import os
# Path for spark source folder
os.environ['SPARK_HOME']="/usr/local/spark-3.1.1-bin-hadoop3.2"
# os.environ['SPARK_LOCAL_IP']="spark://kelvinhwee-VirtualBox:7077"

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
# master = "local[1]"
master = "spark://kelvinhwee-VirtualBox:7077"
config = SparkConf().setAll([('spark.executor.memory', '2g'),
                             ('spark.executor.cores', '2'),
                             ('spark.cores.max', '3'),
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
# rdd = sc.textFile('archive\\testfile.txt')
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
joined_df3 = joined_df2.select(["Movie_ID","Movie_title","Movie_year","Customer_ID","Ratings","Date_of_review","Genre","Duration"])

joined_df3.take(10)

test_pd_df = joined_df3.limit(100).toPandas()