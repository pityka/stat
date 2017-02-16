package stat.sgd.nn

import org.saddle._
import org.saddle.linalg._
import stat.matops._
import stat.sgd._
// abstract out hidden activation, last activation
// regression, multinomial classification
case class NeuralNetwork(shape: Vector[Int],
                         rng: scala.util.Random = scala.util.Random)
    extends ObjectiveFunction[Double, Double] {

  def logistic(x: Double) = {
    val e = math.exp(x)
    e / (1 + e)
  }

  def logisticPrime(x: Double) = {
    val f = logistic(x)
    f * (1 - f)
  }

  val numW = shape.size

  def shapeW(i: Int, xCols: Int): (Int, Int) =
    if (i == 0) (shape(i), xCols)
    else (shape(i), shape(i - 1))

  def extractWs(estimates: Vec[Double], xCols: Int): IndexedSeq[Mat[Double]] = {
    (0 until shape.size)
      .foldLeft((List[Mat[Double]](), 0)) {
        case ((list, idx), i) =>
          val (r, c) = shapeW(i, xCols)
          val copy = Array.ofDim[Double](r * c)
          System.arraycopy(estimates: Array[Double], idx, copy, 0, r * c)
          (new mat.MatDouble(r, c, copy) :: list, idx + r * c)
      }
      ._1
      .reverse
      .toIndexedSeq
  }

  def outputLast(ws: IndexedSeq[Mat[Double]], row: Vec[Double]): Vec[Double] = {
    val penultimate = ws.dropRight(1).foldLeft(row) {
      case (last, w) =>
        val a = w mv last
        a.map(logistic)
    }
    (ws.last mv penultimate) // apply last layer's activation, specific to regression ToDO
  }

  def apply[T: MatOps](b: Vec[Double],
                       batch: Batch[T],
                       work: Option[Array[Double]] = None): Double = {
    val matops = implicitly[MatOps[T]]
    import matops.vops

    val predicted = predict(b, batch.x)

    (predicted - batch.y) vv (predicted - batch.y) * -0.5
  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val matops = implicitly[MatOps[T]]
    import matops.vops

    val ws = extractWs(b, matops.numCols(batch.x))

    matops
      .rows(batch.x)
      .zipWithIndex
      .map {
        case (v, i) =>
          jacobiRow(ws, v.toDense, batch.y.raw(i)) //plug in logistic here
      }
      .reduce(_ + _) * (-1)

  }

  /* Backpropagation as in Shalev-Schwarz, Ben-David page 237 */
  def jacobiRow(ws: IndexedSeq[Mat[Double]],
                row: Vec[Double],
                y: Double): Vec[Double] = {
    val inputs = scala.collection.mutable.ArrayBuffer[Vec[Double]](row)
    val outputs = scala.collection.mutable.ArrayBuffer[Vec[Double]](row)

    /* forward */
    ws.zipWithIndex.foreach {
      case (w, i) =>
        val a = w mv outputs.last
        val o = if (i == ws.size - 1) a else a map logistic

        inputs.append(a)
        outputs.append(o)
    }
    /* backward */
    val dT = outputs.last.map(_ - y) // regression specific stuff here
    val dts = scala.collection.mutable.ArrayBuffer[Vec[Double]](dT)
    val shape1 = row.length +: shape
    val T = shape1.size - 1
    var t = T - 1
    while (t > 0) {
      val w = ws(t)
      val sa = {
        val i = inputs(t + 1)
        if (t == T - 1) i else i.map(logisticPrime)
      }
      val wd = w.mDiagFromLeft(sa)
      val dt = (Mat(dts.last).T mm wd).row(0)
      // val kt = shape1(t - 1)
      // var i = 0
      // val buf = Array.ofDim[Double](kt)
      // while (i < kt) {
      //   var sum = 0d
      //   var j = 0
      //   val ktp1 = shape1(t)
      //   while (j < ktp1) {
      //     sum += ws(t - 1).raw(j, i) * dts.last.raw(j) * logisticPrime(
      //       inputs(t).raw(j))
      //     j += 1
      //   }
      //   buf(i) = sum
      //   i += 1
      // }
      dts.append(dt)
      t -= 1
    }
    dts.append(Vec[Double]()) //placeholder
    val dtsR = dts.reverse
    val jac = Array.ofDim[Double](ws.map(x => x.numRows * x.numCols).sum)
    t = 1
    var c = 0
    while (t <= T) {
      var i = 0
      var I = shape1(t)
      while (i < I) {
        var j = 0
        var J = shape1(t - 1)
        while (j < J) {
          val actP =
            if (t == T - 1) 1d //inputs(t).raw(i) // regression, last layer
            else logisticPrime(inputs(t).raw(i))

          val dp = dtsR(t).raw(i) * actP * outputs(t - 1).raw(j)
          jac(c) = dp
          c += 1
          j += 1
        }
        i += 1
      }
      t += 1
    }
    (jac: Vec[Double])
  }

  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[Double] =
    predict(estimates, data)(DenseMatOps)

  // Specific to regression TODO
  def predict[T: MatOps](estimates: Vec[Double], data: T): Vec[Double] = {
    val matops = implicitly[MatOps[T]]
    import matops.vops

    val ws = extractWs(estimates, matops.numCols(data))

    matops
      .rows(data)
      .map { v =>
        outputLast(ws, v.toDense).raw(0) //plug in logistic here
      }
      .toVec
  }

  //  Specific to regression TODO
  def eval[T: MatOps](est: Vec[Double], batch: Batch[T]): Double = {
    val p = predict(est, batch.x)
    stat.crossvalidation.rSquared(p, batch.y) // plug in logistic here
  }

  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double] =
    start(batch.penalizationMask.length) * 0d + 1

  def start(cols: Int): Vec[Double] = {
    val size = 0 until numW map { i =>
      val (a, b) = shapeW(i, cols); a * b
    } sum

    0 until size map (i => rng.nextDouble) toVec
  }

  def adaptParameterNames(s: Seq[String]): Seq[String] =
    start(s.size).toSeq.zipWithIndex.map(x => "w" + x._2)

  // Don't implement these below, specific to Fista or CoordinateDescent

  def jacobi1D[T: MatOps](b: Vec[Double],
                          batch: Batch[T],
                          i: Int,
                          xmbv: Vec[Double]): Double = ???

  def hessian1D[T: MatOps](p: Vec[Double],
                           batch: Batch[T],
                           i: Int,
                           old: Option[Double],
                           xmvb: Vec[Double]): Double = ???

  def hessian[T: MatOps](p: Vec[Double], batch: Batch[T]): Mat[Double] = ???
  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double = ???

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] = ???
}
