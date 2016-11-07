package stat.sgd

import org.saddle._
import org.saddle.linalg._

case class MultinomialLogisticRegression(numberOfClasses: Int)
    extends ObjectiveFunction[Double, Vec[Double]] {
  assert(numberOfClasses > 1)

  val C = numberOfClasses - 1

  def adaptPenalizationMask(batch: Batch): Vec[Double] =
    mat.repeat(batch.penalizationMask, C).contents

  def adaptParameterNames(s: Seq[String]): Seq[String] = s.flatMap { s =>
    1 until numberOfClasses map { c =>
      s + "_" + c
    }
  }

  def start(cols: Int): Vec[Double] = vec.zeros(cols * C)

  def apply(b: Vec[Double], batch: Batch): Double = {
    val bm = Mat(batch.x.numCols, C, b)

    (0 until batch.x.numRows).map { i =>
      val x: Vec[Double] = batch.x.row(i)
      val y: Double = batch.y.raw(i)
      val denom = {
        ((0 until C) map { (j: Int) =>
          math.exp(x dot bm.col(j))
        }).sum + 1d
      }
      val f =
        if (y == 0d) 0d
        else x dot bm.col(y.toInt - 1)

      f - math.log(denom)
    }.sum

  }

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val bm = Mat(batch.x.numCols, C, b)

    val xp = batch.x mm bm

    val ar = Array.ofDim[Double](b.length)

    var k = 0
    var j = 0
    var i = 0
    var c = 0
    while (k < bm.numRows) {
      while (j < bm.numCols) {
        var s = 0.0
        while (i < batch.y.length) {
          s += jj(k, j, i)
          i += 1
        }
        ar(c) = s
        c += 1
        i = 0
        j += 1
      }
      j = 0
      k += 1
    }

    def jj(k: Int, j: Int, i: Int): Double = {
      val f: Double =
        if (batch.y.raw(i).toInt - 1 == j) batch.x.raw(i, k) else 0.0

      val xpi = xp.row(i)

      val pi = {
        val x = batch.x.row(i)
        val denom = {
          var s = 0d
          var j1 = 0
          while (j1 < C) {
            s += math.exp(xpi.raw(j1))
            j1 += 1
          }
          s + 1d
        }
        val nom = math.exp(xpi.raw(j))
        nom / denom
      }

      f - pi * batch.x.raw(i, k)

    }

    ar

  }

  def hessian(b: Vec[Double], batch: Batch): Mat[Double] = {
    val bm: Mat[Double] = Mat(batch.x.numCols, C, b)

    val xp = batch.x mm bm

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

      val denom = {
        var s = 0d
        var j1 = 0
        while (j1 < C) {
          s += math.exp(xpi.raw(j1))
          j1 += 1
        }
        s + 1d
      }

      val pij = {
        val nom = math.exp(xpi.raw(j))
        nom / denom
      }

      val pijp = {
        val nom = math.exp(xpi.raw(jp))
        nom / denom
      }

      if (j == jp) -1 * xik * xikp * pij * (1 - pij)
      else xik * xikp * pij * pijp

    }

    result
  }

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val h = hessian(p, batch)
    (h * (-1)).eigenValuesSymm(1).raw(0)
  }

  def predict(estimates: Vec[Double], data: Vec[Double]): Vec[Double] = {
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

  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Vec[Double]] =
    Vec(data.rows.map { row =>
      predict(estimates, row)
    }: _*)

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] = ???

  def eval(estimates: Vec[Double], batch: Batch) = {
    val predicted = predict(estimates, batch.x)
    val hard: Vec[Double] = predicted.map { v =>
      val base = 1.0 - v.sum
      (base +: v.toSeq).zipWithIndex.max._2.toDouble
    }

    val accuracy =
      hard.zipMap(batch.y)((p, y) => if (p == y) 1.0 else 0.0).mean

    accuracy
  }

}
