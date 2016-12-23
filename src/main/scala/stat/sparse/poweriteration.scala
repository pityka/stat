package stat.sparse

import stat.matops._
import org.saddle._
import org.saddle.linalg._
import stat.matops.LinearMap._
import slogging.StrictLogging

object Eigen {
  def eigenDecompositionSymmetric[T: LinearMap](
      t: T,
      k: Int,
      maxIter: Int = 100,
      minIter: Int = 5,
      epsilon: Double = 1E-3): EigenDecompositionSymmetric = {

    val start = {
      val r = (0 until t.numCols).toVec.map(_.toDouble)
      r / math.sqrt(r vv r)
    }

    val op = implicitly[LinearMap[T]]

    def loop(t: T,
             k: Int,
             acc: List[(Vec[Double], Double)]): List[(Vec[Double], Double)] =
      if (k == 0) acc
      else {
        val lm = new LinearMap[T] {
          def numCols(t: T) = op.numCols(t)
          def mv(t: T, v: Vec[Double]): Vec[Double] = {
            if (acc.isEmpty) op.mv(t, v)
            else {
              val v1 = op.mv(t, v)
              val v2 = acc.map {
                case (ve, l) =>
                  val m = (Mat(ve) mm Mat(ve).T) * (-1) * l
                  m mv v
              }.reduce(_ + _)
              v1 + v2
            }
          }
        }

        val (v, e, _) =
          PowerIteration.iteration(t, start, maxIter, minIter, epsilon)(lm).get

        loop(t, k - 1, (v -> e) :: acc)

      }

    val evalsEvecs: List[(Vec[Double], Double)] = loop(t, k, Nil).reverse

    val l = evalsEvecs.map(_._2).toVec
    val x = Mat(evalsEvecs.map(_._1): _*)
    EigenDecompositionSymmetric(x, l)

  }
}

object PowerIteration extends StrictLogging {

  def step[T: LinearMap](t: T, b: Vec[Double]) = {
    val tb = t mv b
    val tbl = math.sqrt(tb vv tb)
    val nextVec = tb / tbl
    val nextVal = (b vv tb) / (b vv b)
    (nextVec, nextVal)
  }

  def error[T: LinearMap](t: T, ve: Vec[Double], va: Double): Double = {
    val r = (t mv ve) - (ve * va)
    math.sqrt(r vv r)
  }

  def iteration[T: LinearMap](
      m: T,
      start: Vec[Double],
      max: Int,
      min: Int,
      epsilon: Double): Option[(Vec[Double], Double, Double)] = {
    val t =
      from(m, start).take(max).drop(min).dropWhile(_._3 > epsilon).headOption

    if (t.isEmpty) {
      logger.warn("Did not converge after {} iterations", max)
      None
    } else {
      logger.debug("Converged.")

      t
    }

  }

  def from[T: LinearMap](
      t: T,
      start: Vec[Double]): Stream[(Vec[Double], Double, Double)] = {

    def loop(start: (Vec[Double], Double, Double))
      : Stream[(Vec[Double], Double, Double)] =
      start #:: loop({
        val (ve, _, _) = start
        val (nve, nva) = step(t, ve)

        val co = error(t, nve, nva)

        logger
          .debug("Power iteration error: {}, eval: {}, evec: {}", co, nva, nve)

        (nve, nva, co)
      })

    val (x, y) = step(t, start)
    val e1 = error(t, x, y)
    loop((x, y, e1))
  }

}
