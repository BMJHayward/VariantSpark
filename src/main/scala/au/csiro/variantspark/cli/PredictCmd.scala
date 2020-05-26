package au.csiro.variantspark.cli

import java.io.{FileInputStream, FileOutputStream, ObjectOutputStream, OutputStreamWriter}

import org.json4s.jackson.Serialization.{write, writePretty}
import au.csiro.pbdava.ssparkle.common.arg4j.{AppRunner, TestArgs}
import au.csiro.pbdava.ssparkle.common.utils.{LoanUtils, Logging}
import au.csiro.sparkle.common.args4j.ArgsApp
import au.csiro.variantspark.algo.RandomForestModel
import au.csiro.variantspark.cli.args.FeatureSourceArgs
import au.csiro.variantspark.cmd.EchoUtils._
import au.csiro.variantspark.cmd.Echoable
import au.csiro.variantspark.input._
import au.csiro.variantspark.utils.HdfsPath
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.mllib.regression.LabeledPoint
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

class PredictCmd extends ArgsApp with FeatureSourceArgs with Echoable with Logging with TestArgs {

  @Option(name = "-im", required = true, usage = "Path to input model",
    aliases = Array("--input-model"))
  val inputModel: String = null

  @Option(name = "-of", required = false, usage = "Path to output file (def = stdout)",
    aliases = Array("--output-file"))
  val outputFile: String = null

  @Option(name = "-lf", required = false, usage = "Path to label file",
    aliases = Array("--label-file"))
  val labelFile: String = null

  @Option(name = "-lc", required = false, usage = "Label file column name",
    aliases = Array("--label-column"))
  val labelColumn: String = null

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
    val featureCount = featureSource.features.count.toInt
    val phenoTypes = List("blue", "brown", "black", "green", "yellow", "grey")
    val phenoLabels = Range(0, featureCount).toList
      .map(_ => phenoTypes(Random.nextInt.abs.toInt % phenoTypes.length))
    val phenoLabelIndex = Range(0, featureCount).toList
      .map(_ => Random.nextInt.abs.toDouble % phenoTypes.length)
    val labPts = phenoLabelIndex zip featureSource.features.collect map {
      case (label, feat) => LabeledPoint(label, feat.valueAsVector)
    }
    val labPtsRDD = sc.parallelize(labPts)
    val catInfo = immutable.Map[Int, Int]()
    val numClasses = phenoTypes.length
    val numTrees = 5
    val subsetStrat = "auto"
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val intSeed = 0
    val sparkRFModel = SparkForest.trainClassifier(labPtsRDD, numClasses, catInfo, numTrees,
      subsetStrat, impurity, maxDepth, maxBins, intSeed)
    println("running predict cmd")
    logInfo("Running with params: " + ToStringBuilder.reflectionToString(this))
    echo(s"Analyzing random forest model")
    /*
    if (inputFile != null) {
      val labelSource = new CsvLabelSource(labelFile, labelColumn)
      val labels = labelSource.getLabels(featureSource.sampleNames)
    } else {}
     */
    val rfModel = LoanUtils.withCloseable(new FileInputStream(inputModel)) { in =>
      si.deserializeStream(in).readObject().asInstanceOf[RandomForestModel]
    }
    echo(s"Using spark RF Model: ${sparkRFModel.toString}")
    echo(s"Using labels: ${phenoLabels}")
    echo(s"Loaded rows: ${dumpList(featureSource.sampleNames)}")
    echo(s"Loaded model of size: ${rfModel.size}")
    lazy val featureList =
      featureSource.features.collect().map { feat => (feat.label, feat.valueAsStrings) }
    lazy val inputData = featureSource.features.zipWithIndex().cache()
    val predictions = rfModel.predict(inputData)
    val outputData = featureSource.sampleNames zip predictions
    if (outputFile != null) {
      sc.parallelize(outputData).saveAsTextFile(outputFile)
    } else outputData.foreach(println)
  }
}

object PredictCmd {
  def main(args: Array[String]) {
    AppRunner.mains[PredictCmd](args)
  }
}
