package net.davidrobles.spark

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructType}

object IrisClassifier {

  val IrisDataset = "src/main/resources/iris.data"

  val IrisSchema = new StructType()
    .add("sepal_length", DoubleType)
    .add("sepal_width", DoubleType)
    .add("petal_length", DoubleType)
    .add("petal_width", DoubleType)
    .add("category", StringType)

  def create_pipeline(): Pipeline = {
    val assembler = new VectorAssembler()
      .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
      .setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("category").setOutputCol("label")
    val logisticRegression = new LogisticRegression()
    new Pipeline().setStages(Array(assembler, labelIndexer, logisticRegression))
  }

  def evaluate(model: PipelineModel, dataFrame: DataFrame): Unit = {
    val transformedDF = model.transform(dataFrame)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    println(s"Accuracy: ${evaluator.evaluate(transformedDF)}")
  }

  def train_test_split(irisDF: DataFrame): (DataFrame, DataFrame) = {
    val splits = irisDF.randomSplit(Array(0.8, 0.2))
    (splits(0), splits(1))
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Iris").getOrCreate()
    val irisDF = spark.read.schema(IrisSchema).csv(IrisDataset)
    val (trainDF, testDF) = train_test_split(irisDF)
    val pipeline = create_pipeline()
    val model = pipeline.fit(trainDF)
    evaluate(model, testDF)
  }

}
