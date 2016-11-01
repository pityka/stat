package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait ObjectiveFunction[E] {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double
  def apply(b: Vec[Double], batch: Batch): Double

  def predict(estimates: Vec[Double], data: Vec[Double]): Double
  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Double]

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double]

  def eval(est: Vec[Double], batch: Batch): E
}

object LinearRegression extends ObjectiveFunction[Double] {
  def apply(b: Vec[Double], batch: Batch): Double = {
    val yMinusXb = batch.y - (batch.x mv b)
    (yMinusXb dot yMinusXb) * (-1)
  }

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = y - (X mv b)
    X tmv yMinusXb
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
    (data mv estimates)

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] =
    predict(estimates, data)

  def eval(estimates: Vec[Double], batch: Batch) = {
    val p = predict(estimates, batch.x)
    stat.crossvalidation.rSquared(p, batch.y)
  }

}

object LogisticRegression extends ObjectiveFunction[Double] {
  def apply(b: Vec[Double], batch: Batch): Double = {
    val Xb = (batch.x mv b)
    val yXb = batch.y * Xb
    val z = Xb.map(x => math.log(1d + math.exp(x)))
    (yXb - z).sum
  }

  def getPi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val Xb = (batch.x mv b)
    val eXb = Xb.map(math.exp)
    eXb.map(e => e / (1d + e))
  }

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val pi: Vec[Double] = getPi(b, batch)
    (batch.x tmv (batch.y - pi).col(0))
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
    (data mv estimates).map(math.exp).map(x => x / (1 + x))

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] =
    predict(estimates, data).map(p => if (rng() < p) 1.0 else 0.0)

  def eval(estimates: Vec[Double], batch: Batch) = {
    val p = predict(estimates, batch.x).map(x => if (x > 0.5) 1.0 else 0.0)
    p.zipMap(batch.y)(_ == _).map(x => if (x) 1.0 else 0.0).mean
  }

}
