package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def hessianLargestEigenValue(p: Vec[Double], batch: Batch): Double
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
