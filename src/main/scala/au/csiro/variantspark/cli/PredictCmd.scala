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
import au.csiro.variantspark.utils.HdfsPath
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.JavaSerializer
import org.kohsuke.args4j.Option

class PredictCmd extends ArgsApp with FeatureSourceArgs with Echoable with Logging with TestArgs {

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

  override def run(): Unit = {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration
    println("running predict cmd")
    logInfo("Running with params: " + ToStringBuilder.reflectionToString(this))
    echo(s"Analyzing random forest model")
    val rfModel = LoanUtils.withCloseable(new FileInputStream(inputModel)) { in =>
      si.deserializeStream(in).readObject().asInstanceOf[RandomForestModel]
    }
    echo(s"Loaded rows: ${dumpList(featureSource.sampleNames)}")
    echo(s"Loaded model of size: ${rfModel.size}")
    // want feature.label and feature.value
    lazy val featureList =
      featureSource.features.collect().map { feat => (feat.label, feat.valueAsStrings) }
    featureSource.sampleNames zip featureList foreach println
    featureSource.features.map { feat =>
      (feat.label, feat.valueAsStrings, feat.valueAsStrings.length)
    } foreach println

    // rfModel.printout
    lazy val inputData = featureSource.features.zipWithIndex().cache()
    val predictions = rfModel.predict(inputData)
    val outputData = featureSource.sampleNames zip predictions
    if (outputFile != null) {
      // rdd = sc.parallelize(outputData)
      // try rdd.take(100).foreach(println) to print
      sc.parallelize(outputData).saveAsTextFile(outputFile)
    } else outputData.foreach(println)
  }
}

object PredictCmd {
  def main(args: Array[String]) {
    AppRunner.mains[PredictCmd](args)
  }
}
