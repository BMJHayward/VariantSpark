package au.csiro.variantspark.external

import au.csiro.variantspark.algo.{
  DecisionTreeModel,
  DecisionTreeNode,
  LeafNode,
  RandomForestMember,
  RandomForestModel,
  RandomForestParams,
  SplitNode
}
import org.json4s.{DefaultFormats, JsonAST, _}
import org.json4s.JsonDSL._

class TreeSerializer
    extends CustomSerializer[Node](format =>
        ({
        case obj: JObject =>
          implicit val formats: Formats = format

          if ((obj \ "left") == JArray(List()) || (obj \ "right") == JArray(List())) {
            // stuff
            obj.extract[Leaf]
          } else {
            // other stuff
            obj.extract[Split]
          }
      }, {
        case split: SplitNode =>
          implicit val formats: Formats = format
          JObject("majlab" -> JInt(split.majorityLabel),
            "clscount" -> JArray(List(split.classCounts.toList)), "size" -> JInt(split.size),
            "ndimp" -> JDouble(split.nodeImpurity), "splitvar" -> JLong(split.splitVariableIndex),
            "splitpt" -> JDouble(split.splitPoint), "impred" -> JDouble(split.impurityReduction),
            "left" -> Extraction.decompose(split.left),
            "right" -> Extraction.decompose(split.right), "isperm" -> JBool(split.isPermutated))
        case leaf: LeafNode =>
          implicit val formats: Formats = format
          JObject("majlab" -> JInt(leaf.majorityLabel),
            "clscount" -> JArray(List(leaf.classCounts.toList)), "size" -> JInt(leaf.size),
            "ndimp" -> JDouble(leaf.nodeImpurity))
      }))

class ModelConverter(varIndex: Map[Long, String]) {

  def toExternal(node: DecisionTreeNode): Node = {
    node match {
      case LeafNode(majorityLabel, classCounts, size, nodeImpurity) =>
        Leaf(majorityLabel, classCounts, size, nodeImpurity)
      case SplitNode(majorityLabel, classCounts, size, nodeImpurity, splitVariableIndex,
          splitPoint, impurityReduction, left, right, isPermutated) =>
        Split(majorityLabel, classCounts, size, nodeImpurity,
          varIndex.getOrElse(splitVariableIndex, null), splitVariableIndex, isPermutated,
          splitPoint, impurityReduction, toExternal(left), toExternal(right))
      case _ => throw new IllegalArgumentException("Unknow node type:" + node)
    }
  }

  def toExternal(rfMember: RandomForestMember): Tree = {
    val rootNode = rfMember.predictor match {
      case DecisionTreeModel(rootNode) => toExternal(rootNode)
      case _ => throw new IllegalArgumentException("Unknow predictory type:" + rfMember.predictor)
    }
    Tree(rootNode,
      Option(rfMember.oobIndexes).map(_ => OOBInfo(rfMember.oobIndexes, rfMember.oobPred)))
  }

  def toExternal(rfModel: RandomForestModel): Forest = {
    val oobErrors =
      if (rfModel.oobErrors != null && rfModel.oobErrors.nonEmpty
          && !rfModel.oobErrors.head.isNaN) {
        Some(rfModel.oobErrors)
      } else {
        None
      }
    Forest(Option(rfModel.params), rfModel.members.map(toExternal), oobErrors)
  }

  def toInternalNode(node: JsonAST.JValue): Node = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    node match {
      case JObject(List(("left", _))) => node.extract[Leaf]
      case JObject(List(("right", _))) => node.extract[Leaf]
      case JObject(List(("left", _))) => node.extract[Split]
      case JObject(List(("right", _))) => node.extract[Split]
      case JObject(List(("rootNode", _))) => node.extract[Split]
      case _ => throw new IllegalArgumentException("Unknown node type:" + node)
    }
  }

  def toInternal(inTree: JsonAST.JValue): Node = {
    // crazy times. try a customer deserialiser e.g.
    // https://stackoverflow.com/questions/54322448/how-to-deserialize-a-scala-tree-with-json4s

    /*
    class TreeSerializer extends CustomSerializer[Tree](format => ({
      case obj: JObject =>
        implicit val formats: Formats = format

        if ((obj \ "trees") == JNothing) {
          Leaf(
            (obj \ "nameL").extract[String]
          )
        } else {
          Node(
            (obj \ "nameN").extract[String],
            (obj \ "trees").extract[List[Tree]]
          )
        }
    }, {
      case node: Node =>
        JObject("nameN" -> JString(node.nameN), "trees" -> node.trees.map(Extraction.decompose))
      case leaf: Leaf =>
        "nameL" -> leaf.nameL
    }))
    implicit val formats = DefaultFormats + new TreeSerializer
    read[Tree](tree)
     */
    // tree.children.map(toInternalNode(_).asInstanceOf[RandomForestMember])
    implicit val formats: Formats = DefaultFormats + new TreeSerializer
    toInternalNode(inTree)
  }

  def toInternalForest(forest: JsonAST.JValue, labelCount: Int): List[Node] = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    val forestChildren: List[Node] = forest.children.map { tree => toInternal(tree) }
    forestChildren
  }
}
