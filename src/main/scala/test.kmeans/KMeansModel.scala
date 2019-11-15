package test.kmeans

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{SparseVector, Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.slf4j.{Logger, LoggerFactory}

object KMeansModel {
  private val logger: Logger = LoggerFactory.getLogger(KMeansModel.getClass)

}

class KMeansModel(val clusterCenters: Array[OldVector]) {
  @transient protected val logger: Logger = KMeansModel.logger
  private val clusterCentersWithNorm =
    if (clusterCenters == null) null else clusterCenters.map(new VectorWithNorm(_))
  final val dimensions = 999990
  //set iteration's time
  val iterations = 5
  //set K
  val k = 3
  //set random
  val randomItem = 12345
  //set epsilon
  val epsilon = 1e-4
  //set feature's precision
  val precision = 1e-6

  //load data
  def loadData(path: String): Dataset[_]= {
    val conf = new SparkConf()
      .setAppName("KMeans Test")
    //      .setMaster("local")
    //      .set("spark.executor.memory","1g")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    // Import Kmeans clustering Algorithm
    // Load the Wholesale Customers Data
    val startTime = System.currentTimeMillis()
    val data = spark
      .read
      .option("header","true")
      .option("inferschema","true")
      .format("csv")
      .load("Wholesale customers data.csv")

    val totalLength = data.count()
    val feature_data = (data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
    val training_data = assembler.transform(feature_data).select("features")
    logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
    println(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
    training_data
  }

  def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data.takeSample(false, k, new XORShiftRandom(this.randomItem).nextInt())
      .map(_.vector).distinct.map(new VectorWithNorm(_))
  }

  def changeDataForm(data :Dataset[_]): RDD[OldVector] ={
    val instances: RDD[OldVector] = data.select("features").rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }
    instances
  }


  def run(
           data: Dataset[_]
         ): KMeansModel ={
    //Change data form from dataframe[_] to rdd[vector]
    val instances: RDD[OldVector] = data.select("features").rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }
    if (instances.getStorageLevel == StorageLevel.NONE){
      logger.info(s"The input data is not directly cached,which may hurt performance if its" + "parent RDDs are also uncached.")
    }
    //compute squared norms and cache them
    val norms = instances.map(OldVectors.norm(_, 2.0))
    norms.persist()
    val zippedData = instances.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v,norm)
    }
    val model = runAlgorithm(zippedData)
    norms.unpersist()
    model
  }

  def dot(
           v1:OldVector,
           v2:OldVector
         ): Double = {
    val vec1 = v1.toArray
    val vec2 = v2.toArray
    var ans = 0.0
    for(i <- 0 to vec1.length){
      ans += vec1(i) * vec2(i)
    }
    ans
  }

  def fastSquaredDistance(
                           v1: VectorWithNorm,
                           v2: VectorWithNorm): Double = {
    val vector1 = v1.vector
    val norm1 = v1.norm
    val vector2 = v2.vector
    val norm2 = v2.norm
    val n = vector1.size
    require(vector2.size == n)
    require(norm1 >= 0.0 && norm2 >= 0.0)
    val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
    val normDiff = norm1 - norm2
    var sqDist = 0.0
    /*
     * The relative error is
     * <pre>
     * EPSILON * ( \|a\|_2^2 + \|b\\_2^2 + 2 |a^T b|) / ( \|a - b\|_2^2 ),
     * </pre>
     * which is bounded by
     * <pre>
     * 2.0 * EPSILON * ( \|a\|_2^2 + \|b\|_2^2 ) / ( (\|a\|_2 - \|b\|_2)^2 ).
     * </pre>
     * The bound doesn't need the inner product, so we can use it as a sufficient condition to
     * check quickly whether the inner product approach is accurate.
     */
    val precisionBound1 = 2.0 * epsilon * sumSquaredNorm / (normDiff * normDiff + epsilon)
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * dot(vector1, vector2)
    } else if (vector1.isInstanceOf[SparseVector] || vector2.isInstanceOf[SparseVector]) {
      val dotValue = dot(vector1, vector2)
      sqDist = math.max(sumSquaredNorm - 2.0 * dotValue, 0.0)
      val precisionBound2 = epsilon * (sumSquaredNorm + 2.0 * math.abs(dotValue)) /
        (sqDist + epsilon)
      if (precisionBound2 > precision) {
        sqDist = OldVectors.sqdist(vector1, vector2)
      }
    } else {
      sqDist = OldVectors.sqdist(vector1, vector2)
    }
    sqDist
  }

  //???Whether the dimension could be used like this
  def axpy(figure:Double,
           x:OldVector,
           y:OldVector): Unit ={
    var yArray = y.toArray
    var xArray = x.toArray
    for(i <- 0 until yArray.length){
      yArray(i) += xArray(i)
    }
    y = OldVectors.dense(yArray)
  }

  def scal(figure:Double,
           vec:OldVector): Unit ={
    for (i <- 0 until vec.size){
      vec(i) = (1.0)/figure * vec(i)
    }
  }


  def findClosest(
                   centers: TraversableOnce[VectorWithNorm],
                   point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  def runAlgorithm(data: RDD[VectorWithNorm]): KMeansModel = {
    val sc = data.sparkContext
    val initStartTime = System.currentTimeMillis()
    val centers = initRandom(data)
    val nowTime = System.currentTimeMillis()
    logger.info(s"Initialization took ${nowTime - initStartTime} ms")

    //whether converged
    var converged = false
    var cost = 0.0
    var iteration = 0
    val iterationStartTime = System.currentTimeMillis()

    //Execute iterations until converged
    while(iteration < iterations && !converged){
      val costAccum = sc.doubleAccumulator
      val bcCenters = sc.broadcast(centers)

      //Find new centers
      val newCenters = data.mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.vector.size

        val sums = Array.fill(thisCenters.length)(OldVectors.zeros(dims))
        val counts = Array.fill(thisCenters.length)(0L)

        points.foreach { point =>
          val (bestCenter, cost) = findClosest(thisCenters, point)
          costAccum.add(cost)
          val sum = sums(bestCenter)
          axpy(1.0, point.vector, sum)
          counts(bestCenter) += 1
        }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.mapValues { case (sum, count) =>
        scal(1.0 / count, sum)
        new VectorWithNorm(sum)
      }.collectAsMap()

      bcCenters.destroy()

      // Update the cluster centers and costs
      converged = true
      newCenters.foreach { case (j, newCenter) =>
        if (converged && fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {
          converged = false
        }
        centers(j) = newCenter
      }

      cost = costAccum.value
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logger.info(s"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == iterations) {
      logger.info(s"KMeans reached the max number of iterations: ${iterations}.")
    } else {
      logger.info(s"KMeans converged in $iteration iterations.")
    }

    logger.info(s"The cost is $cost.")

    new KMeansModel(centers.map(_.vector))
  }

  def computeCost(data: RDD[OldVector]): Double = {
    val bcCentersWithNorm = data.context.broadcast(clusterCentersWithNorm)
    val cost = data
      .map(p => findClosest(bcCentersWithNorm.value, new VectorWithNorm(p))).collect()
    var total = 0.0
    for(i <- 0 until cost.length){
      total += cost.apply(i)._2
    }
    println(s"Withing set sum of squared errors = $total")
    bcCentersWithNorm.destroy()
    total
  }


}

//Used to calculate distance
class VectorWithNorm(val vector: OldVector, val norm: Double) extends Serializable {
//  def this(vector: OldVector,norm:Double) = this(vector,norm)

  def this(vector: OldVector) = this(vector, OldVectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(OldVectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(OldVectors.dense(vector.toArray), norm)
}

