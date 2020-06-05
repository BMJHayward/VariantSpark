package au.csiro.variantspark.cli

import au.csiro.pbdava.ssparkle.common.arg4j.{AppRunner, TestArgs}
import au.csiro.pbdava.ssparkle.common.utils.Logging
import au.csiro.sparkle.common.args4j.ArgsApp
import au.csiro.variantspark.cli.args.{
  FeatureSourceArgs,
  LabelSourceArgs,
  ModelOutputArgs,
  RandomForestArgs
}
import au.csiro.variantspark.cmd.EchoUtils._
import au.csiro.variantspark.cmd.Echoable
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.hadoop.conf.Configuration
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{RandomForest => SparkForest}
import org.apache.spark.serializer.JavaSerializer
import org.kohsuke.args4j.Option

import scala.collection._
import scala.util.Random

class TrainRFCmd
    extends ArgsApp with LabelSourceArgs with RandomForestArgs with FeatureSourceArgs
    with ModelOutputArgs with Echoable with Logging with TestArgs {

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

  override def run(): Unit = {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration
    val featureCount = featureSource.features.count.toInt
    val phenoTypes = List("blue", "brown", "black", "green", "yellow", "grey")
    val phenoLabels = Range(0, featureCount).toList
      .map(_ => phenoTypes(Random.nextInt.abs % phenoTypes.length))
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
    println("running train cmd")
    logInfo("Running with params: " + ToStringBuilder.reflectionToString(this))
    echo(s"Analyzing random forest model")
    echo(s"Using spark RF Model: ${sparkRFModel.toString}")
    echo(s"Using labels: $phenoTypes")
    echo(s"Loaded rows: ${dumpList(featureSource.sampleNames)}")

    if (modelFile != null) {
      sparkRFModel.save(sc, modelFile)
    }
  }
}

object TrainRFCmd {
  def main(args: Array[String]) {
    AppRunner.mains[TrainRFCmd](args)
  }
}
