package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait Penalty[H] {
  def proximal(b: Vec[Double], batch: Batch, alpha: Double): Vec[Double]
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def apply(b: Vec[Double], batch: Batch): Double
  def withHyperParameter(h: H): Penalty[H]
}

case class L2(lambda: Double) extends Penalty[Double] {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    b * batch.penalizationMask * lambda

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    mat.diag(batch.penalizationMask * lambda)

  def proximal(w: Vec[Double], batch: Batch, alpha: Double): Vec[Double] =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      old / (1.0 + pw * lambda * alpha))

  def apply(b: Vec[Double], batch: Batch): Double =
    (b.map(x => x * x) * batch.penalizationMask * 0.5 * lambda).sum

  def withHyperParameter(h: Double) = L2(h)
}

case class L1(lambda: Double) extends Penalty[Double] {

  def proximal(w: Vec[Double], batch: Batch, alpha: Double) =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      math.signum(old) * math.max(0.0, math.abs(old) - lambda * alpha * pw))

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    throw new RuntimeException("L1 jacobi")

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    throw new RuntimeException("L1 hessian")

  def apply(b: Vec[Double], batch: Batch): Double =
    (b.map(math.abs) * batch.penalizationMask * 0.5 * lambda).sum

  def withHyperParameter(h: Double) = L1(h)

}

case class ElasticNet(lambda1: Double, lambda2: Double)
    extends Penalty[(Double, Double)] {

  def proximal(w: Vec[Double], batch: Batch, alpha: Double) =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      if (pw.isPosInfinity) 0.0
      else
        math.signum(old) * math.max(
          0.0,
          (math
            .abs(old) - lambda1 * alpha * pw) / (1.0 + pw * alpha * lambda2)))

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    throw new RuntimeException("EN Jacobi")

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    throw new RuntimeException("EN Hessian")

  def apply(b: Vec[Double], batch: Batch): Double =
    (b.map(math.abs) * batch.penalizationMask * 0.5 * lambda1).sum +
      (b.map(x => x * x) * batch.penalizationMask * 0.5 * lambda2).sum

  def withHyperParameter(h: (Double, Double)) = ElasticNet(h._1, h._2)

}
