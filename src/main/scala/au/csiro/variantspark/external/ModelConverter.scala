package au.csiro.variantspark.external

import au.csiro.variantspark.algo.{
  DecisionTreeModel,
  DecisionTreeNode,
  LeafNode,
  RandomForestMember,
  RandomForestModel,
  SplitNode
}

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

  def toInternal(leafOrSplit: Node): DecisionTreeNode with Product = {
    leafOrSplit match {
      case Leaf(majorityLabel, classCounts, size, nodeImpurity) =>
        LeafNode(majorityLabel, classCounts, size, nodeImpurity)
      case Split(majorityLabel, classCounts, size, nodeImpurity, splitVar, splitVarIndex,
          isPermutated, splitPoint, impurityReduction, left, right) =>
        SplitNode(majorityLabel, classCounts, size, nodeImpurity,
          splitVarIndex.toString + splitVar, splitPoint, impurityReduction, toInternal(left),
          toInternal(right), isPermutated)
      case _ => throw new IllegalArgumentException("Unknown node type:" + leafOrSplit)
    }

  }

  def toInternal(tree: Tree): RandomForestMember = { tree =>
    // crazy times. try a customer deserialiser e.g.
    // https://stackoverflow.com/questions/54322448/how-to-deserialize-a-scala-tree-with-json4s
    tree.asInstanceOf[RandomForestMember]
  }

  def toInternal(forest: Forest, labelCount: Int): RandomForestModel = {
    RandomForestModel(forest.trees.map(toInternal), labelCount, forest.oobErrors,
      Some(forest.params))
  }

}
