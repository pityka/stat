package stat.sgd

import org.saddle._
import org.saddle.linalg._

case class MultinomialLogisticRegression(numberOfClasses: Int)
    extends ObjectiveFunction[Double] {
  assert(numberOfClasses > 1)

  val C = numberOfClasses - 1

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

    0 until bm.numCols flatMap { j =>
      0 until bm.numRows map { k =>
        ((0 until batch.y.length) map { i =>
          val f: Double =
            if (batch.y.raw(i).toInt - 1 == j) batch.x.raw(i, k) else 0.0

          val pi = {
            val x = batch.x.row(i)
            val denom = {

              (0 until C map { j =>
                math.exp(x dot bm.col(j))
              }).sum + 1d
            }
            val nom = math.exp(x dot bm.col(j))
            nom / denom
          }

          f - pi * batch.x.raw(i, k)
        }).sum

      }
    } toVec

  }

  def hessian(b: Vec[Double], batch: Batch): Mat[Double] = {
    val bm = Mat(batch.x.numCols, C, b)

    val vec = 0 until bm.numCols flatMap { j =>
      0 until bm.numRows flatMap { k =>
        0 until bm.numCols flatMap { jp =>
          0 until bm.numRows map { kp =>
            (0 until batch.y.length map { i =>
              val xik = batch.x.raw(i, k)
              val xikp = batch.x.raw(i, kp)

              val pij = {
                val x = batch.x.row(i)
                val denom = {

                  (0 until C map { j =>
                    math.exp(x dot bm.col(j))
                  }).sum + 1d
                }
                val nom = math.exp(x dot bm.col(j))
                nom / denom
              }

              val pijp = {
                val x = batch.x.row(i)
                val denom = {

                  (0 until C map { j =>
                    math.exp(x dot bm.col(j))
                  }).sum + 1d
                }
                val nom = math.exp(x dot bm.col(jp))
                nom / denom
              }

              if (j == jp) -1 * xik * xikp * pij * (1 - pij)
              else xik * xikp * pij * pijp

            }).sum

          }
        }
      }
    } toArray

    Mat(b.length, b.length, vec)
  }

  def minusHessianLargestEigenValue(p: Vec[Double], batch: Batch): Double = {
    val h = hessian(p, batch)
    (h * (-1)).eigenValuesSymm(1).raw(0)
  }

  def predict(estimates: Vec[Double], data: Vec[Double]): Double = ???

  def predict(estimates: Vec[Double], data: Mat[Double]): Vec[Double] = ???

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double] = ???

  def eval(estimates: Vec[Double], batch: Batch) = ???

}
