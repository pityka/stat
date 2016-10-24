package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double
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

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val s = batch.x.singularValues(1).raw(0)
    s * s
  }
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
    batch.x tmm Mat(batch.x.cols.zip(w.toSeq).map {
      case (col, w) => col * w * (-1)
    }: _*)
  }

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val pi = getPi(p, batch)
    val w: Vec[Double] = (pi * (pi * (-1) + 1.0)).map(x => math.sqrt(x))
    val wx =
      if (batch.x.numRows < batch.x.numCols)
        Mat(batch.x.rows.zip(w.toSeq).map {
          case (col, w) => col * w * (-1)
        }: _*)
      else
        Mat(batch.x.cols.zip(w.toSeq).map {
          case (col, w) => col * w * (-1)
        }: _*)

    val s = wx.singularValues(1).raw(0)
    s * s
  }
}
