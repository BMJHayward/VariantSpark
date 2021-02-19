// import JSON branch
import java.io.{FileInputStream, ObjectOutputStream, OutputStreamWriter}
import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.JavaSerializer
import au.csiro.pbdava.ssparkle.common.utils.LoanUtils
import scala.io.Source
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.{read, write, writePretty}
import org.json4s.jackson.Serialization
import org.json4s.JsonAST.{JObject, JValue}
import au.csiro.variantspark.external._
import au.csiro.variantspark.algo.{
  DecisionTree,
  DecisionTreeModel,
  DecisionTreeNode,
  Impurity,
  LeafNode,
  RandomForest,
  RandomForestMember,
  RandomForestModel,
  RandomForestParams,
  SplitImpurity,
  SplitInfo,
  SplitNode,
  TreeFeature,
  impurity,
  split
}
import au.csiro.variantspark.utils.HdfsPath
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf

val conf = new SparkConf(!System.getProperty("sparkle.local", "false").toBoolean).setAppName(getClass.getSimpleName)
if (conf.contains("spark.master")) conf else conf.setMaster("local")

def loadModel(inputModel: String, inputModelFormat: String) = inputModelFormat match {
  case "json" => spark.read.json(inputModel).asInstanceOf[RandomForestModel]
  case "ser" | "java" | "bin" | "model" | "spark" => {
    val javaSerializer = new JavaSerializer(conf)
    val si = javaSerializer.newInstance()
    LoanUtils.withCloseable(new FileInputStream(inputModel)) { in =>
      si.deserializeStream(in).readObject().asInstanceOf[RandomForestModel]
    }}
  case _ => sc.textFile(inputModel)
}

//implicit val jsonFormats = DefaultFormats
implicit val formats = DefaultFormats

val src = Source.fromFile("data/ch22-model.json").mkString
val mj = parse(src, useBigDecimalForDouble = true) // parse from json4s.jackson
val params = mj.children(0)
val forest = mj.children(1)
val md = forest \ "rootNode" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left"
val nd = md \ "left"
val od = md \ "right"

// have a look at what we've got
println(pretty(render(mj)))
mj.values // show Map of everything
// parse the AST in mj to return RandomForestModel
forest \ "rootNode"
forest \ "rootNode" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left" \ "left"

def getForest(node: JsonAST.JValue): Unit = {
  @scala.annotation.tailrec
  def inner(node: JsonAST.JValue): Unit = {
    println(pretty(render(node)))
    val innernode = (node \ "left")
    if (innerNode == JsonAST.JNothing) {
      inner(node \ "right")
    }
  }
  inner(node)
}


def getTree(tree: JsonAST.JValue): Unit = tree.children match {
  case JObject(List(("left", LeafNode))) => println("got a node")
  case JObject(List(("right", LeafNode))) => println("got a node")
  case JObject(List(("left", SplitNode))) => println("got a node")
  case JObject(List(("right", SplitNode))) => println("got a node")
  case JObject(List(("rootNode", SplitNode))) => println("got a node")
  case _ => println("oops")
}

def toInternal(inModel){
match {
  case Leaf => something()
  case Split => something()
  case Tree => something()
  case Forest => something()
  case Node => something()
  case params => something()
  case trees => something()
  case left => something()
  case right => something()
  case _ => something()
}

}

def saveModelJson(rfModel: RandomForestModel, outputModel: String) {
  implicit val hadoopConf: Configuration = sc.hadoopConfiguration
  implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
  LoanUtils.withCloseable(new OutputStreamWriter(HdfsPath(outputModel).create())) { objectOut =>
    writePretty(rfModel, objectOut)
  }
}


def loadModelJson(inputModel: String) {
  implicit val hadoopConf: Configuration = sc.hadoopConfiguration
  implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
  LoanUtils.withCloseable(new FileInputStream(inputModel)) { objectIn =>
    Serialization.read(objectIn).asInstanceOf(RandomForestModel)
  }
}

