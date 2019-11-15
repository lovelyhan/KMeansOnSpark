package test.kmeans

import org.apache.log4j._
import org.apache.spark.{SparkConf, SparkContext}

object KMeansApp {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    val model = new KMeansModel(null)
    val path = "/user/hadoop/data/WholesaleCustomersData.csv"
    val data = model.loadData(path)
    val updateModel = model.run(data)
    //  // Fit that model to the training_data
    val RDD = updateModel.changeDataForm(data)
    val WSSSE = updateModel.computeCost(RDD)
  }
}
