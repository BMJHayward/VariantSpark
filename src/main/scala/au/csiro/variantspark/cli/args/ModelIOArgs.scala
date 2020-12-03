package au.csiro.variantspark.cli.args
import java.io.{FileInputStream, ObjectOutputStream, OutputStreamWriter}

import au.csiro.pbdava.ssparkle.common.utils.LoanUtils
import au.csiro.variantspark.algo.RandomForestModel
import au.csiro.variantspark.cmd.Echoable
import au.csiro.variantspark.external.ModelConverter
import au.csiro.variantspark.utils.HdfsPath
import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.JavaSerializer
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{NoTypeHints, _}
import org.kohsuke.args4j.{Option => ArgsOption}

import scala.io.Source

trait ModelIOArgs extends SparkArgs with Echoable {

  @ArgsOption(name = "-im", required = false, usage = "Path to model file",
    aliases = Array("--input-model"))
  val inputModel: String = null

  @ArgsOption(name = "-imf", required = false,
    usage = "Format of the model file, one of: `json`, `java` (def=`java`)",
    aliases = Array("--input-model-format"))
  val inputModelFormat: String = "java"

  @ArgsOption(name = "-om", required = false, usage = "Path to model file",
    aliases = Array("--output-model"))
  val outputModel: String = null

  @ArgsOption(name = "-omf", required = false,
    usage = "Format of the model file, one of: `json`, `java` (def=`java`)",
    aliases = Array("--output-model-format"))
  val outputModelFormat: String = "java"

  def requiresFullIndex: Boolean = inputModel != null

  def loadModel(inputModel: String, inputModelFormat: String): RandomForestModel = {
    inputModelFormat match {
      case "json" => loadModelJson(inputModel)
      case "ser" | "java" | "bin" | "model" | "spark" => loadModelJava(inputModel)
      case _ =>
        throw new IllegalArgumentException("Unrecognised model format: " + inputModelFormat)
    }
  }

  def loadModelJson(inputModel: String): RandomForestModel = {
    implicit val formats = Serialization.formats(NoTypeHints).preservingEmptyValues
    val src = Source.fromFile(inputModel).mkString
    val model = parse(src)
    val params = model.children(0)
    val forest = model.children(1)
    val labelCount = 3
    val someThing = new ModelConverter(Map(1L -> "1.0")).toInternal(forest, labelCount)
    someThing
    /*
    LoanUtils.withCloseable(new FileInputStream(inputModel)) { in =>
      parseJson(file2JsonInput(in))
      Serialization.read[RandomForestModel](in)
    }
    val src = scala.io.Source.fromFile(inputModel)
    val lines = {
      try src.mkString
      finally src.close()
    }
    parse(lines)
    val modelString = sc.textFile(inputModel)
    val modelJ: RDD[JsonInput => JValue] = modelString.map(_ => parse(_, true, true))
   */
  }

  def loadModelJava(inputModel: String): RandomForestModel = {
    val javaSerializer = new JavaSerializer(conf)
    val si = javaSerializer.newInstance()
    LoanUtils.withCloseable(new FileInputStream(inputModel)) { in =>
      si.deserializeStream(in).readObject().asInstanceOf[RandomForestModel]
    }
  }

  def saveModel(rfModel: RandomForestModel, variableIndex: Map[Long, String]) {
    if (outputModel != null) {
      echo(s"Saving random forest model as `${outputModelFormat}` to: ${outputModel}")
      outputModelFormat match {
        case "java" => saveModelJava(rfModel, variableIndex)
        case "json" => saveModelJson(rfModel, variableIndex)
        case _ =>
          throw new IllegalArgumentException(
              "Unrecognized model format type: " + outputModelFormat)
      }

    }
  }

  def saveModelJson(rfModel: RandomForestModel, variableIndex: Map[Long, String]) {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration
    implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
    LoanUtils.withCloseable(new OutputStreamWriter(HdfsPath(outputModel).create())) { objectOut =>
      writePretty(new ModelConverter(variableIndex).toExternal(rfModel), objectOut)
    }
  }

  def saveModelJava(rfModel: RandomForestModel, variableIndex: Map[Long, String]) {
    implicit val hadoopConf: Configuration = sc.hadoopConfiguration
    LoanUtils.withCloseable(new ObjectOutputStream(HdfsPath(outputModel).create())) { objectOut =>
      objectOut.writeObject(rfModel)
    }
  }
}
