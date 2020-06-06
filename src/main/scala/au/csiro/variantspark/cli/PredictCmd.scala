package au.csiro.variantspark.cli

import java.io.{FileInputStream, FileOutputStream, ObjectOutputStream, OutputStreamWriter}

import org.json4s.jackson.Serialization.{write, writePretty}
import au.csiro.pbdava.ssparkle.common.arg4j.{AppRunner, TestArgs}
import au.csiro.pbdava.ssparkle.common.utils.{LoanUtils, Logging}
import au.csiro.sparkle.common.args4j.ArgsApp
import au.csiro.variantspark.algo.RandomForestModel
import au.csiro.variantspark.cli.args.{
  FeatureSourceArgs,
  LabelSourceArgs,
  ModelOutputArgs,
  RandomForestArgs
}
import au.csiro.variantspark.cmd.EchoUtils._
import au.csiro.variantspark.cmd.Echoable
import au.csiro.variantspark.input._
import au.csiro.variantspark.utils.HdfsPath
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{RandomForestModel => SparkForestModel}
import org.apache.spark.mllib.tree.{
  DecisionTree,
  GradientBoostedTrees,
  RandomForest => SparkForest
}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.JavaSerializer
import org.kohsuke.args4j.Option
import scala.collection._
import scala.util.Random

class PredictCmd
    extends ArgsApp with FeatureSourceArgs with LabelSourceArgs with Echoable with Logging
    with TestArgs {

  @Option(name = "-im", required = true, usage = "Path to input model",
    aliases = Array("--input-model"))
  val inputModel: String = null

  @Option(name = "-of", required = false, usage = "Path to output file (def = stdout)",
    aliases = Array("--output-file"))
  val outputFile: String = null
  val javaSerializer = new JavaSerializer(conf)
  val si = javaSerializer.newInstance()

  override def testArgs: Array[String] =
    Array("-im", "file.model", "-if", "file.data", "-of", "outputpredictions.file")

  def percCalc(M: Map[Int, Int]): Map[Int, Float] = {
    val total = M.values.sum
    Map(0 -> M.getOrElse(0, 0).toFloat / total, 1 -> M.getOrElse(1, 0).toFloat / total,
      2 -> M.getOrElse(2, 0).toFloat / total)
  }

  override def run(): Unit = {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration

    // echo(s"Loading labels from: ${featuresFile}, column: ${featureColumn}")
    if (featuresFile != null) {
      val labFile =
        spark.read.format("csv").option("header", "true").load(featuresFile)
      val labs = labFile.select(featureColumn).rdd.map(_(0)).collect.toList
      echo(s"Loaded labels from file: ${labs.toSet}")
    } else {
      // val labelSource = new CsvLabelSource(featuresFile, featureColumn)
      // val labels = labelSource.getLabels(featureSource.sampleNames)
      val labels = List("blue", "brown", "black", "green", "yellow", "grey")
      echo(s"Loaded labels: ${dumpList(labels.toList)}")
    }

    val featureCount = featureSource.features.count.toInt
    val phenoTypes = List("blue", "brown", "black", "green", "yellow", "grey")
    val phenoLabels = Range(0, featureCount).toList
      .map(_ => phenoTypes(Random.nextInt.abs % phenoTypes.length))

    println("running predict cmd")
    logInfo("Running with params: " + ToStringBuilder.reflectionToString(this))
    echo(s"Analyzing random forest model")
    val sparkRFModel = SparkForestModel.load(sc, inputModel)
    val labelMap = sc.textFile(inputModel + ".labelMap")

    echo(s"Using spark RF Model: ${sparkRFModel.toString}")
    echo(s"Loaded rows: ${dumpList(featureSource.sampleNames)}")
    echo(s"Trees in model: ${sparkRFModel.numTrees}")
    echo(s"Nodes in model: ${sparkRFModel.totalNumNodes}")

    lazy val featureVectors = featureSource.features.map { f => f.valueAsVector }
    val predictions = {
      sparkRFModel.predict(featureVectors)
    }

    if (outputFile != null) {
      predictions.saveAsTextFile(outputFile)
    } else predictions.foreach(println)
    // } else predictions.map{p => (p._1, labels(p._2)} foreach(println)
  }
}

object PredictCmd {
  def main(args: Array[String]) {
    AppRunner.mains[PredictCmd](args)
  }
}
