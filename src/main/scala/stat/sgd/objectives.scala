package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

trait ObjectiveFunction[E, @specialized(Double) P] {

  def jacobi1D[T: MatOps](b: Vec[Double],
                          batch: Batch[T],
                          i: Int,
                          xmbv: Vec[Double]): Double

  def hessian1D[T: MatOps](p: Vec[Double],
                           batch: Batch[T],
                           i: Int,
                           old: Option[Double],
                           xmvb: Vec[Double]): Double
  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double]
  def hessian[T: MatOps](p: Vec[Double], batch: Batch[T]): Mat[Double]
  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double
  def apply[T: MatOps](b: Vec[Double],
                       batch: Batch[T],
                       work: Option[Array[Double]] = None): Double
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

  def apply[T: MatOps](b: Vec[Double],
                       batch: Batch[T],
                       work: Option[Array[Double]]): Double = {
    val xmvb =
      if (work.isEmpty) (batch.x mv b)
      else
        (batch.x.mv(b, work.get))

    var s = 0d
    var j = 0
    val n = batch.y.length
    val y = batch.y
    while (j < n) {
      val t = (y.raw(j) - xmvb.raw(j))
      s += t * t
      j += 1
    }

    s * (-1d)
  }

  def jacobi1D[T: MatOps](b: Vec[Double],
                          batch: Batch[T],
                          i: Int,
                          XmvB: Vec[Double]): Double = {
    val matop = implicitly[MatOps[T]]
    import matop.vops
    val y = batch.y
    val X = batch.x

    /*  matop.col(batch.x, i) vv y - XmvB */

    var s = 0d
    var j = 0
    val n = y.length
    while (j < n) {
      s += matop.raw(batch.x, j, i) * (y.raw(j) - XmvB.raw(j))
      j += 1
    }
    s
    val r = s
    // assert(r == jacobi(b, batch).raw(i), r + " " + jacobi(b, batch).raw(i))
    r
  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = y - (X mv b)
    X tmv yMinusXb
  }

  def hessian1D[T: MatOps](p: Vec[Double],
                           batch: Batch[T],
                           i: Int,
                           old: Option[Double],
                           xmvb: Vec[Double]): Double = {
    if (old.isDefined && batch.full) old.get
    else {
      val matop = implicitly[MatOps[T]]
      import matop.vops
      // val c = matop.col(batch.x, i)
      // (c vv2 c) * (-1)
      var s = 0d
      var j = 0
      val n = batch.y.length
      while (j < n) {
        s += matop.raw(batch.x, j, i) * matop.raw(batch.x, j, i)
        j += 1
      }
      s * (-1)
    }
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

  def apply[T: MatOps](b: Vec[Double],
                       batch: Batch[T],
                       work: Option[Array[Double]]): Double = {
    val Xb = (batch.x mv b)
    val yXb = batch.y * Xb
    val z = Xb.map(x => math.log(1d + math.exp(x)))
    (yXb - z).sum
  }

  def getPi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val Xb = (batch.x mv b)
    Xb.map { x =>
      val e = math.exp(x)
      e / (1 + e)
    }
  }

  def jacobi1D[T: MatOps](b: Vec[Double],
                          batch: Batch[T],
                          i: Int,
                          XmvB: Vec[Double]): Double = {
    val matop = implicitly[MatOps[T]]
    import matop.vops
    val y = batch.y
    val X = batch.x
    val c = matop.col(batch.x, i)

    val pi = XmvB.map { x =>
      val e = math.exp(x)
      e / (1 + e)
    }

    /* c vv y - XmvB */

    val r = vops.vv(c, y - pi)
    // assert(r == jacobi(b, batch).raw(i), r + " " + jacobi(b, batch).raw(i))
    r
  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val pi: Vec[Double] = getPi(b, batch)
    (batch.x tmv (batch.y - pi).col(0))
  }

  def hessian1D[T: MatOps](p: Vec[Double],
                           batch: Batch[T],
                           i: Int,
                           old: Option[Double],
                           xmvb: Vec[Double]): Double = {
    val matop = implicitly[MatOps[T]]
    import matop.vops
    val c = matop.col(batch.x, i)
    val pi: Vec[Double] = xmvb.map { x =>
      val e = math.exp(x)
      e / (1 + e)
    }
    val w: Vec[Double] = pi * (pi * (-1) + 1.0)
    ((c * (w * (-1))) vv2 c)

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
