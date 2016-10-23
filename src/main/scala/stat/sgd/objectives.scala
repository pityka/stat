package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def hessianLargestEigenValue(p: Vec[Double], batch: Batch): Double
  def apply(b: Vec[Double], batch: Batch): Double
}

trait Penalty {
  def proximal(b: Vec[Double], batch: Batch, alpha: Double): Vec[Double]
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def apply(b: Vec[Double], batch: Batch): Double
}

object LinearRegression extends ObjectiveFunction {
  def apply(b: Vec[Double], batch: Batch): Double = {
    val yMinusXb = batch.y - (batch.x mm Mat(b)).col(0)
    (yMinusXb dot yMinusXb) * (-1)
  }

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = Mat(y) - (X mm Mat(b))
    (yMinusXb tmm X).row(0)
  }

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] = {
    batch.x.innerM * (-1)
  }

  def hessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val s = batch.x.singularValues(1).raw(0)
    s * s
  }
}

case class L2(lambda: Double) extends Penalty {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    b * batch.penalizationMask * lambda

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    mat.diag(batch.penalizationMask * lambda)

  def proximal(w: Vec[Double], batch: Batch, alpha: Double): Vec[Double] =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      old / (1.0 + pw * lambda * alpha))

  def apply(b: Vec[Double], batch: Batch): Double =
    (b.map(x => x * x) * batch.penalizationMask * 0.5 * lambda).sum
}

case class L1(lambda: Double) extends Penalty {

  def proximal(w: Vec[Double], batch: Batch, alpha: Double) =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      math.signum(old) * math.max(0.0, math.abs(old) - lambda * alpha * pw))

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    throw new RuntimeException("L1 jacobi")
  // b.zipMap(batch.penalizationMask)((b, p) =>
  // (if (b == 0) 0d else math.signum(b) * lambda * p))

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    throw new RuntimeException("L1 hessian")
  // mat.zeros(p.length, p.length)

  def apply(b: Vec[Double], batch: Batch): Double =
    (b.map(math.abs) * batch.penalizationMask * 0.5 * lambda).sum

}
