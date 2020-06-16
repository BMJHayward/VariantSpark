package au.csiro.variantspark.cli

import au.csiro.pbdava.ssparkle.common.arg4j.{AppRunner, TestArgs}
import au.csiro.pbdava.ssparkle.common.utils.Logging
import au.csiro.sparkle.common.args4j.ArgsApp
import au.csiro.variantspark.cli.args.{FeatureSourceArgs, LabelSourceArgs}
import au.csiro.variantspark.cmd.EchoUtils._
import au.csiro.variantspark.cmd.Echoable
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.hadoop.conf.Configuration
import org.apache.spark.mllib.tree.model.{RandomForestModel => SparkForestModel}
import org.apache.spark.serializer.{JavaSerializer, SerializerInstance}
import org.kohsuke.args4j.Option

import scala.collection._

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
  val si: SerializerInstance = javaSerializer.newInstance()

  override def testArgs: Array[String] =
    Array("-im", "file.model", "-if", "file.data", "-of", "outputpredictions.file")

  def percCalc(M: Map[Int, Int]): Map[Int, Float] = {
    val total = M.values.sum
    Map(0 -> M.getOrElse(0, 0).toFloat / total, 1 -> M.getOrElse(1, 0).toFloat / total,
      2 -> M.getOrElse(2, 0).toFloat / total)
  }

  override def run(): Unit = {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration

    println("running predict cmd")
    logInfo("Running with params: " + ToStringBuilder.reflectionToString(this))
    echo(s"Analyzing random forest model")
    val sparkRFModel = SparkForestModel.load(sc, inputModel)
    val labelMap = sc.textFile(inputModel + ".labelMap")
    val newMap = labelMap
      .map { l => l.stripPrefix("(").stripSuffix(")").split(",") }
      .map { t => (t(0).toInt, t(1)) }
      .collect
      .toMap

    echo(s"Using spark RF Model: ${sparkRFModel.toString}")
    echo(s"Loaded rows: ${dumpList(featureSource.sampleNames)}")
    echo(s"Trees in model: ${sparkRFModel.numTrees}")
    echo(s"Nodes in model: ${sparkRFModel.totalNumNodes}")
    echo(s"Labels in labelMap:\n${labelMap foreach println}")

    lazy val featureVectors = featureSource.features.map { f => f.valueAsVector }
    val predictions = {
      sparkRFModel.predict(featureVectors)
    }
    if (outputFile != null) {
      predictions.saveAsTextFile(outputFile)
      predictions
        .map { p: Double => (newMap(p.toInt), p) }
        .saveAsTextFile(outputFile + ".labelMap")
    } else predictions.foreach(println)
    echo(s"Label, Count, Frequency")
    predictions.countByValue.map { p =>
      (p._1, p._2, p._2 / predictions.count.toDouble)
    } foreach println
    // } else predictions.map{p => (p._1, labels(p._2)} foreach(println)
  }
}

object PredictCmd {
  def main(args: Array[String]) {
    AppRunner.mains[PredictCmd](args)
  }
}
