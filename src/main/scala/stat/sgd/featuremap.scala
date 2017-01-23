package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{Prediction, NamedPrediction}
import slogging.StrictLogging
import stat.matops._
import stat.io.upicklers._
import upickle.default._
import upickle.Js
import stat.rbf.RadialBasisFunctions._

trait FeatureMapFactory[H] {
  def apply(h: H): FeatureMap
}

object IdentityFeatureMapFactory extends FeatureMapFactory[Double] {
  def apply(h: Double) = IdentityFeatureMap
}

sealed trait FeatureMap {
  def applyMat[T: MatOps](b: T): T
  def applyPenalizationMask(v: Vec[Double]): Vec[Double]
}

object IdentityFeatureMap extends FeatureMap {
  def applyMat[T: MatOps](b: T): T = b
  def applyPenalizationMask(v: Vec[Double]): Vec[Double] = v
}

case class RbfFeatureMapFactory(centers: Seq[(Vec[Double], Double)])
    extends FeatureMapFactory[Double] {
  def apply(h: Double) = RbfFeatureMap(centers.map(x => x._1 -> x._2 * h))
}

case class RbfFeatureMap(centers: Seq[(Vec[Double], Double)])
    extends FeatureMap {
  def applyMat[T: MatOps](b: T): T = {
    val top = implicitly[MatOps[T]]
    import top.vops

    def euclid(v1: Vec[Double], v2: top.V, v1inner: Double): Double =
      vops.euclid(v2, v1, v1inner)

    def makeRow(data: top.V,
                centers: Seq[(Vec[Double], Double, Double)]): top.V =
      top.vops.fromElems((1d +: centers.map(center =>
        gaussian(center._2)(euclid(center._1, data, center._3)))).toVec)

    val centersWithInner = centers.map(x => (x._1, x._2, x._1 vv x._1))

    top.fromRows(top.rows(b).map { row =>
      makeRow(row, centersWithInner) append row
    })

  }

  def applyPenalizationMask(v: Vec[Double]): Vec[Double] =
    Vec(1d +: vec.ones(centers.size + v.length).toSeq: _*)

}
