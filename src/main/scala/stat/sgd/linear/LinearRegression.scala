package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

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

}
