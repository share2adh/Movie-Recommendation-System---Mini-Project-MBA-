#!/usr/bin/env python
# coding: utf-8

# In[22]:


## ENTER THE SOURCE FILE DIRECTORY
file_path = 'C:/Users/Adh/Desktop/Mini_Proj/Data/ratings_small.csv'


first_recommendation = Movie_Recommender(file_path)
first_recommendation.show()


def Movie_Recommender(file_path):
    
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import SparkSession 
    
    spark = SparkSession.builder.getOrCreate()
    
    df = spark.read.csv(file_path, inferSchema=True, header=True)

    data = df.selectExpr("cast(userId as int) userId",
        "cast(movieId as int) movieId",
        "cast(rating as int) rating")
    

    (training, test) = data.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    print("Root-Mean-square error = " + str(rmse))
    userRecs = model.recommendForAllUsers(10)
    return userRecs



# In[ ]:




