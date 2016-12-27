package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

trait ObjectiveFunction[E, @specialized(Double) P] {
  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double]
  def hessian[T: MatOps](p: Vec[Double], batch: Batch[T]): Mat[Double]
  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double
  def apply[T: MatOps](b: Vec[Double], batch: Batch[T]): Double
  def start(cols: Int): Vec[Double]

  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[P]
  def predict[T: MatOps](estimates: Vec[Double], data: T): Vec[P]

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double]

  def eval[T: MatOps](est: Vec[Double], batch: Batch[T]): E

  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double]
  def adaptParameterNames(s: Seq[String]): Seq[String]
  def scaleBackCoefficients(estimates: Vec[Double],
                            std: Vec[Double]): Vec[Double]
}

object LinearRegression extends ObjectiveFunction[Double, Double] {
  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double] =
    batch.penalizationMask
  def adaptParameterNames(s: Seq[String]): Seq[String] = s

  def start(cols: Int): Vec[Double] = vec.zeros(cols)

  def apply[T: MatOps](b: Vec[Double], batch: Batch[T]): Double = {
    val yMinusXb = batch.y - (batch.x mv b)
    (yMinusXb vv yMinusXb) * (-1)
  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = y - (X mv b)
    X tmv yMinusXb
  }

  def hessian[T: MatOps](p: Vec[Double], batch: Batch[T]): Mat[Double] = {
    batch.x.innerM * (-1)
  }

  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double = {
    val s = batch.x.singularValues(1).raw(0)
    s * s
  }
  def predict(estimates: Vec[Double], data: Vec[Double]): Double =
    estimates dot data
  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[Double] =
    (data mv estimates)
  def predict[T: MatOps](estimates: Vec[Double], data: T): Vec[Double] =
    (data mv estimates)

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] =
    predictMat(estimates, data)

  def eval[T: MatOps](estimates: Vec[Double], batch: Batch[T]) = {
    val p = predict(estimates, batch.x)
    stat.crossvalidation.rSquared(p, batch.y)
  }

  def scaleBackCoefficients(estimates: Vec[Double], std: Vec[Double]) =
    estimates * std

}

object LogisticRegression
    extends ObjectiveFunction[(Double, Double, Double, Int), Double] {
  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double] =
    batch.penalizationMask
  def adaptParameterNames(s: Seq[String]): Seq[String] = s

  def start(cols: Int): Vec[Double] = vec.zeros(cols)

  def apply[T: MatOps](b: Vec[Double], batch: Batch[T]): Double = {
    val Xb = (batch.x mv b)
    val yXb = batch.y * Xb
    val z = Xb.map(x => math.log(1d + math.exp(x)))
    (yXb - z).sum
  }

  def getPi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val Xb = (batch.x mv b)
    val eXb = Xb.map(math.exp)
    eXb.map(e => e / (1d + e))
  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val pi: Vec[Double] = getPi(b, batch)
    (batch.x tmv (batch.y - pi).col(0))
  }

  def hessian[T: MatOps](b: Vec[Double], batch: Batch[T]): Mat[Double] = {
    val pi = getPi(b, batch)
    val w: Vec[Double] = pi * (pi * (-1) + 1.0)
    batch.x.mDiagFromLeft(w * (-1)) tmm batch.x
  }

  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double = {
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
  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[Double] =
    (data mv estimates).map(math.exp).map(x => x / (1 + x))

  def predict[T: MatOps](estimates: Vec[Double], data: T): Vec[Double] =
    (data mv estimates).map(math.exp).map(x => x / (1 + x))

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] =
    predictMat(estimates, data).map(p => if (rng() < p) 1.0 else 0.0)

  def eval[T: MatOps](estimates: Vec[Double], batch: Batch[T]) = {
    val p = predict(estimates, batch.x).map(x => if (x > 0.5) 1.0 else 0.0)
    val accuracy = p.zipMap(batch.y)(_ == _).map(x => if (x) 1.0 else 0.0).mean

    val tp = p
      .zipMap(batch.y)((p, y) => p == 1d && y == 1d)
      .map(x => if (x) 1.0 else 0.0)
      .sum

    val fp =
      p.zipMap(batch.y)((p, y) => p == 1d && y == 0d)
        .map(x => if (x) 1.0 else 0.0)
        .sum

    val precision = tp / (tp + fp)
    val recall = tp / batch.y.sum

    (accuracy, precision, recall, batch.y.length)
  }

  def scaleBackCoefficients(estimates: Vec[Double], std: Vec[Double]) =
    estimates * std

}
