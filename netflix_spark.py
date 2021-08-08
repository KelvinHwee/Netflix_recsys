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
from pyspark.sql.functions import isnan, when, count, col, array, create_map
from pyspark.sql.functions import max, min
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
joined_df = netflix_spark_df2.join(movie_titles_spark_df, ["Movie_ID"]) # put "join" condition in list drops duplicates
joined_df2 = joined_df.join(IMDB_spark_df, movie_titles_spark_df["Movie_title"] == IMDB_spark_df["Title"])
joined_df3 = joined_df2.select(["Movie_ID","Movie_title","Movie_year",
                                "Customer_ID","Ratings","Date_of_review",
                                "Genre","Duration"])


#=== create the rows for the matrix setup (to determine user similarity)

#--- we check out a sub-portion of the dataframe
joined_df3.createOrReplaceTempView("movie_table") # create a view in order for Spark SQL to work
ratings_df = spark.sql(
                        """
                        SELECT Movie_ID, Customer_ID, Ratings FROM movie_table
                        WHERE year(Date_of_review) >= 2001  --total 37,021,254 entries; ratings_df.select("Ratings").count()
                        ORDER BY Date_of_review DESC  
                        """
                    )

ratings_df.show(20)


#--- create a dictionary that computes an index number for the "Customer_ID" (for matrix computation later)

#- create a UDF that applies the created dictionary, and then do the mapping
def mapping_expr(dict_object):
    def mapping_expr_(value):
        return dict_object.get(value)
    return udf(mapping_expr_, IntegerType())

#- get a list of movie IDs
distinct_movie_id_rdd = joined_df3.select("Movie_ID").distinct().rdd
distinct_movie_id_list = distinct_movie_id_rdd.map(lambda r: r[0]).collect()
distinct_movie_id_list2 = sorted(distinct_movie_id_list)

#- get list of customer IDs
distinct_cust_id_rdd = ratings_df.select("Customer_ID").distinct().rdd
distinct_cust_id_list = distinct_cust_id_rdd.map(lambda r: r[0]).collect()
distinct_cust_id_list2 = sorted(distinct_cust_id_list)

#- create the dictionaries
cust_id_dict = dict((cust, idx) for idx, cust in enumerate(distinct_cust_id_list2))
movie_id_dict = dict((mov, idx) for idx, mov in enumerate(distinct_movie_id_list2))

#- insert a new column for the customer ID index
# max value of movie idx 3745
# max value of cust idx  475502
ratings_df2 = ratings_df.withColumn("CustID_idx", mapping_expr(cust_id_dict)(col("Customer_ID"))) \
                        .withColumn("MovieID_idx", mapping_expr(movie_id_dict)(col("Movie_ID")))

ratings_df2.show(10)
ratings_df2.printSchema()


### TESTING AREA
ratings_subset_df = ratings_df2.filter(ratings_df2.CustID_idx > 475000) \
                               .filter(ratings_df2.MovieID_idx > 3100)

ratings_subset_df.show()
ratings_subset_df.count()


#--- get matrix of ratings against customers
# rating = ratings_subset_df.where((col("CustID_idx") == 3235) & (col("MovieID_idx") == 475458)).select("Ratings")
rating = ratings_subset_df.where(col("CustID_idx") > 3234).select("Ratings")
rating = ratings_subset_df.where(col("CustID_idx") == 3562)
rating = ratings_subset_df.filter(ratings_subset_df.MovieID_idx == '3562')
# rating.show()


master_ratings_list = []
for cust_num in ratings_subset_df.select("CustID_idx"):
    for movie_num in ratings_subset_df.select("MovieID_idx"):

        ratings_list = []
        rating = ratings_subset_df.filter((col("CustID_idx") == cust_num) & (col("MovieID_idx") == movie_num)).select("Ratings")

        if rating.count() > 0:
            ratings_list.append(rating)
        elif rating.count() == 0:
            ratings_list.append(0)

    master_ratings_list.append(ratings_list)

type(master_ratings_list)
master_ratings_list.show()


# movies_list = []
# ratings_list = []
# for cust_num in range(ratings_df.select("CustID_idx").count()):
#     sub_movie_index_list = ratings_df2.filter("CustID_idx == " + str(cust_num)).select("MovieID_idx").collect()
#     sub_ratings_list = ratings_df2.filter("CustID_idx == " + str(cust_num)).select("Ratings").collect()
#
#     movies_list.append(sub_movie_index_list)
#     ratings_list.append(sub_ratings_list)

    # sub_movie_index_list = ratings_df2.filter("CustID_idx == " + str(i)).select("MovieID_idx").collect()



#-


#=== create the customer/movie/ratings matrix

# col_headers = ["cust_id"] + distinct_movie_id_list2 # this will be used as col headers for the ratings dataframe
# row_headers = ["cust_id_" + str(i) for i in distinct_cust_id_list2]







#
# par_x = sc.parallelize([1,2,3,4,5])
# type(par_x)
# par_x.collect()
#
# df = spark.createDataFrame(
#     sc.parallelize([["id" + str(n)] + np.random.randint(0, 2, 10).tolist() for n in range(20)]),
#     ["id"] + ["movie" + str(i) for i in range(10)])
# df.show()


### TESTING AREA


#=== perform recommendation by determining similar users
#=== this approach however, suffers from the "cold start problem"