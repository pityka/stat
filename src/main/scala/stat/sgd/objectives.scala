package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double
  def apply(b: Vec[Double], batch: Batch): Double

  def predict(estimates: Vec[Double], data: Vec[Double]): Double
  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Double]
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

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val s = batch.x.singularValues(1).raw(0)
    s * s
  }
  def predict(estimates: Vec[Double], data: Vec[Double]): Double =
    estimates dot data
  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Double] =
    (data mm estimates).col(0)
}

object LogisticRegression extends ObjectiveFunction {
  def apply(b: Vec[Double], batch: Batch): Double = {
    val Xb = (batch.x mm Mat(b)).col(0)
    val yXb = batch.y * Xb
    val z = Xb.map(x => math.log(1d + math.exp(x)))
    (yXb - z).sum
  }

  def getPi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val Xb = (batch.x mm Mat(b)).col(0)
    val eXb = Xb.map(math.exp)
    eXb.map(e => e / (1d + e))
  }

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val pi: Vec[Double] = getPi(b, batch)
    (batch.x tmm (batch.y - pi)).col(0)
  }

  def hessian(b: Vec[Double], batch: Batch): Mat[Double] = {
    val pi = getPi(b, batch)
    val w: Vec[Double] = pi * (pi * (-1) + 1.0)
    batch.x.mDiagFromLeft(w * (-1)) tmm batch.x
  }

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val pi = getPi(p, batch)
    val w: Vec[Double] = (pi * (pi * (-1) + 1.0)).map(x => math.sqrt(x))
    val wx =
      batch.x.mDiagFromLeft(w * (-1))

    val s = wx.singularValues(1).raw(0)
    s * s
  }

  def predict(estimates: Vec[Double], data: Vec[Double]): Double = {
    val e = math.exp(estimates dot data)
    e / (1 + e)
  }
  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Double] =
    (data mm estimates).col(0).map(math.exp).map(x => x / (1 + x))
}
