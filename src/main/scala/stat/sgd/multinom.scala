package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._
case class MultinomialLogisticRegression(numberOfClasses: Int)
    extends ObjectiveFunction[(Double, Int), Vec[Double]] {
  assert(numberOfClasses > 1)

  val C = numberOfClasses - 1

  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double] =
    mat.repeat(batch.penalizationMask, C).contents

  def adaptParameterNames(s: Seq[String]): Seq[String] = s.flatMap { s =>
    1 until numberOfClasses map { c =>
      s + "_" + c
    }
  }

  def start(cols: Int): Vec[Double] = vec.zeros(cols * C)

  def apply[T: MatOps](b: Vec[Double], batch: Batch[T]): Double = {
    val bm = Mat(batch.x.numCols, C, b)

    val xp = (batch.x mm bm).map(math.exp)

    (0 until batch.x.numRows).foldLeft(0d) { (sum, i) =>
      val x: Vec[Double] = batch.x.row(i)
      val y: Double = batch.y.raw(i)

      val xpi = xp.row(i)

      val denom = xpi.sum + 1

      val f =
        if (y == 0d) 0d
        else (x vv bm.col(y.toInt - 1))

      sum + f - math.log(denom)
    }

  }

  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double] = {
    val bm = Mat(batch.x.numCols, C, b)

    val xp = (batch.x mm bm).map(math.exp)

    Mat(((0 until C) map { j =>
      val y1 = batch.y.map { y =>
        if (y.toInt - 1 == j) 1d else 0d
      }

      val pi = Array.ofDim[Double](batch.y.length)
      var i = 0
      while (i < pi.length) {

        val xpi = xp.row(i)
        val xpis = xpi.sum

        val denom =
          xpis + 1d

        val nom = (xpi.raw(j))
        pi(i) = nom / denom

        i += 1
      }

      val t = y1 - (pi: Vec[Double])

      (batch.x tmv t).col(0)

    }): _*).contents

  }

  // def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
  //   val bm = Mat(batch.x.numCols, C, b)
  //
  //   val xp = (batch.x mm bm).map(math.exp)
  //
  //   val ar = Array.ofDim[Double](b.length)
  //
  //   var k = 0
  //   var j = 0
  //   var i = 0
  //   var c = 0
  //   while (k < bm.numRows) {
  //     while (j < bm.numCols) {
  //       var s = 0.0
  //       while (i < batch.y.length) {
  //         s += jj(k, j, i)
  //         i += 1
  //       }
  //       ar(c) = s
  //       c += 1
  //       i = 0
  //       j += 1
  //     }
  //     j = 0
  //     k += 1
  //   }
  //
  //   def jj(k: Int, j: Int, i: Int): Double = {
  //     val f: Double =
  //       if (batch.y.raw(i).toInt - 1 == j) batch.x.raw(i, k) else 0.0
  //
  //     val xpi = xp.row(i)
  //
  //     val pi = {
  //       val x = batch.x.row(i)
  //       val denom = {
  //         var s = 0d
  //         var j1 = 0
  //         while (j1 < C) {
  //           s += (xpi.raw(j1))
  //           j1 += 1
  //         }
  //         s + 1d
  //       }
  //       val nom = (xpi.raw(j))
  //       nom / denom
  //     }
  //
  //     f - pi * batch.x.raw(i, k)
  //
  //   }
  //   ar
  // }

  def hessian[T: MatOps](b: Vec[Double], batch: Batch[T]): Mat[Double] = {
    val bm: Mat[Double] = Mat(batch.x.numCols, C, b)

    val xp = (batch.x mm bm).map(math.exp)

    val result = mat.zeros(b.length, b.length)
    var ar = result.contents

    var k = 0
    var j = 0
    var kp = 0
    var jp = 0
    var i = 0
    var c = 0

    while (k < bm.numRows) {
      while (j < bm.numCols) {
        while (kp < bm.numRows) {
          while (jp < bm.numCols) {
            var s = 0d
            while (i < batch.y.length) {
              val v = h(k, j, kp, jp, i)
              s += v
              i += 1
            }
            ar(c) = s
            c += 1
            i = 0
            jp += 1
          }
          jp = 0
          kp += 1
        }
        kp = 0
        j += 1
      }
      j = 0
      k += 1
    }

    def h(k: Int, j: Int, kp: Int, jp: Int, i: Int) = {
      val xik = batch.x.raw(i, k)
      val xikp = batch.x.raw(i, kp)
      val x: Vec[Double] = batch.x.row(i)
      val xpi = xp.row(i)
      val xpis = xpi.sum

      val denom =
        xpis + 1d

      val pij = {
        val nom = (xpi.raw(j))
        nom / denom
      }

      val pijp = {
        val nom = (xpi.raw(jp))
        nom / denom
      }

      if (j == jp) -1 * xik * xikp * pij * (1 - pij)
      else xik * xikp * pij * pijp

    }
    result
  }

  /**
    * This is used only to set the minimum step size
    * This does not return the correct value but hopefully something close to it
    */
  def minusHessianLargestEigenValue[T: MatOps](b: Vec[Double],
                                               batch: Batch[T]): Double = {

    val bm: Mat[Double] = Mat(batch.x.numCols, C, b)

    val xp = (batch.x mm bm).map(math.exp)

    val x2 = {
      var ar = Array.ofDim[Double](batch.y.length * (C) * batch.x.numCols)
      var i = 0
      var j = 0
      var k = 0
      var c = 0
      while (i < batch.y.length) {
        val xpi = xp.row(i)
        val denom = xpi.sum + 1
        while (k < batch.x.numCols) {
          val x = batch.x.raw(i, k)
          while (j < C) {
            val pij = xpi.raw(j) / denom
            ar(c) = pij * (-1) * x
            c += 1
            j += 1
          }
          j = 0
          k += 1
        }
        k = 0
        i += 1
      }
      Mat(batch.y.length, (C) * batch.x.numCols, ar)
    }

    val x3 = {
      var ar = Array.ofDim[Double](batch.y.length * (C) * batch.x.numCols)
      var i = 0
      var j = 0
      var k = 0
      var c = 0
      while (i < batch.y.length) {
        val xpi = xp.row(i)
        val denom = xpi.sum + 1
        while (k < batch.x.numCols) {
          val x = batch.x.raw(i, k)
          while (j < C) {
            val pij = xpi.raw(j) / denom
            ar(c) = math.sqrt(pij * (1 - pij)) * (-1) * x
            c += 1
            j += 1
          }
          j = 0
          k += 1
        }
        k = 0
        i += 1
      }
      Mat(batch.y.length, (C) * batch.x.numCols, ar)
    }

    val s1 = x2.singularValues(1).raw(0)
    val s2 = x3.singularValues(1).raw(0)

    s1 * s1 + s2 * s2

    // val h = hessian(p, batch)
    // (h * (-1)).eigenValuesSymm(1).raw(0)
  }

  def predictVec[V: VecOps](estimates: Vec[Double], data: V): Vec[Double] = {
    val bm = Mat(data.length, C, estimates)

    val denom = {
      (0 until C map { j =>
        math.exp(data dot bm.col(j))
      }).sum + 1d
    }

    val vec =
      0 until bm.numCols map { j =>
        val nom = math.exp(data dot bm.col(j))
        nom / denom
      } toVec

    vec
  }

  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[Vec[Double]] =
    Vec(data.rows.map { row =>
      predictVec(estimates, row)(DenseVecOps)
    }: _*)

  def predict[T](estimates: Vec[Double], data: T)(
      implicit m: MatOps[T]): Vec[Vec[Double]] = {
    // val p = new Pimp(data)(m)
    // implicitly[m.V =:= p.V]
    Vec(m.rows(data).map { row =>
      predictVec(estimates, row)
    }: _*)
  }

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] = ???

  def eval[T: MatOps](estimates: Vec[Double], batch: Batch[T]) = {
    val predicted = predict(estimates, batch.x)
    val hard: Vec[Double] = predicted.map { v =>
      val base = 1.0 - v.sum
      (base +: v.toSeq).zipWithIndex.max._2.toDouble
    }

    val accuracy =
      hard.zipMap(batch.y)((p, y) => if (p == y) 1.0 else 0.0).mean

    (accuracy, batch.y.length)
  }

}
