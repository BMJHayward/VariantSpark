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
import org.json4s
import org.json4s.{DefaultFormats, JsonAST, _}
import org.json4s.JsonDSL._

class TreeSerializer
    extends CustomSerializer[Node](
        format =>
          ({
          case obj: JObject =>
            implicit val formats = {
              format
            }

            if ((obj \ "left") == JArray(List()) || (obj \ "right") == JArray(List())) {
              // stuff
              obj.extract[Leaf]
            } else {
              // other stuff
              obj.extract[Split]
            }
        }, {
          case split: SplitNode =>
            JObject("majlab" -> JString(split.majorityLabel),
              "clscount" -> JString(split.classCounts), "size" -> JString(split.size),
              "ndimp" -> JString(split.nodeImpurity),
              "splitvar" -> JString(split.splitVariableIndex),
              "splitpt" -> JString(split.splitPoint),
              "impred" -> JString(split.impurityReduction), "left" -> JString(split.left),
              "right" -> JString(split.right), "isperm" -> JString(split.isPermutated))
          case leaf: LeafNode =>
            JObject("majlab" -> JString(leaf.majorityLabel),
              "clscount" -> JString(leaf.classCounts), "size" -> JString(leaf.size),
              "ndimp" -> JString(leaf.nodeImpurity))
        }))

class ModelConverter(varIndex: Map[Long, String]) {

  def toExternal(node: DecisionTreeNode): Node = {
    node match {
      case LeafNode(majorityLabel, classCounts, size, nodeImpurity) =>
        Leaf(majorityLabel, classCounts, size, nodeImpurity)
      case SplitNode(majorityLabel, classCounts, size, nodeImpurity, splitVariableIndex,
          splitPoint, impurityReduction, left, right, isPermutated) => {
        Split(majorityLabel, classCounts, size, nodeImpurity,
          varIndex.getOrElse(splitVariableIndex, null), splitVariableIndex, isPermutated,
          splitPoint, impurityReduction, toExternal(left), toExternal(right))
      }
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
    implicit val formats = DefaultFormats
    node match {
      case JObject(List(("left", _))) => node.extract[Leaf]
      case JObject(List(("right", _))) => node.extract[Leaf]
      case JObject(List(("left", _))) => node.extract[Split]
      case JObject(List(("right", _))) => node.extract[Split]
      case JObject(List(("rootNode", _))) => node.extract[Split]
      case _ => throw new IllegalArgumentException("Unknown node type:" + node)
    }
  }

  def toInternal(tree: Tree): RandomForestMember = { tree =>
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

    tree.asInstanceOf[RandomForestMember]
    tree.children.map(toInternalNode(_).asInstanceOf[RandomForestMember])
  }

  def toInternalForest(forest: JsonAST.JValue, labelCount: Int): RandomForestModel = {
    implicit val formats = DefaultFormats
    val forestChildren: List[RandomForestMember] = forest.children.map { tree =>
      toInternalTree(tree)
    }.toList
    RandomForestModel(forestChildren, labelCount,
      forest.extract[List[Double]], forest.extract[RandomForestParams])
  }
}
