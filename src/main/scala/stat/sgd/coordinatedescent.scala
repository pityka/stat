package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._
import stat.matops._

case class CDState(point: Vec[Double],
                   convergence: Double,
                   obj: Double,
                   hessianDiag: Vec[Double])
    extends ItState

object CoordinateDescentUpdater extends Updater[CDState] {
  def next[M: MatOps](x: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[CDState]): CDState = {

    val penalizationMask = obj.adaptPenalizationMask(batch)

    val innerSteps = 1

    val hessianDiag = last.map(_.hessianDiag).getOrElse {
      0 until x.length map { i =>
        obj.hessian1D(x, batch, i, None)
      } toVec
    }

    def objective(w: Vec[Double]): Double =
      objectiveUnpenalized(w) + penaltyFunction(w)

    def penaltyFunction(w: Vec[Double]): Double =
      pen.apply(w, penalizationMask)

    def shrink(w: Double, alpha1: Double, pm: Double) =
      pen.proximal1D(w, pm, alpha1)

    def objectiveUnpenalized(w: Vec[Double]): Double = -1 * obj.apply(w, batch)

    /* 1D Newton, then shrink. */
    def updateCoordinate(i: Int, x: Vec[Double], xmvb: Vec[Double]) = {
      val stepSize = {
        val s = (-1d) / obj.hessian1D(x, batch, i, Some(hessianDiag.raw(i)))
        if (s.isInfinite) 1E-16 else s
      }
      val gradient = obj.jacobi1D(x, batch, i, xmvb) * (-1)
      shrink(x.raw(i) - gradient * stepSize, stepSize, penalizationMask.raw(i))
    }

    val objCurrent = last.map(_.obj).getOrElse(objective(x))

    val matops = implicitly[MatOps[M]]

    // mutated
    val xmvb: Array[Double] = batch.x mv x

    // mutated
    val x2: Array[Double] = {
      val ar = Array.ofDim[Double](x.length)
      System.arraycopy((x: Array[Double]), 0, ar, 0, x.length)
      ar
    }

    var i = 0
    val N = x.length
    while (i < N) {

      val xcol: Array[Double] =
        (matops.col(batch.x, i)).asInstanceOf[Vec[Double]] // TODO

      var K = innerSteps
      while (K > 0) {
        val v = updateCoordinate(i, x2, xmvb)

        var j = 0
        val n = xmvb.size
        while (j < n) {
          xmvb(j) += xcol(j) * (v - x2(i))
          j += 1
        }

        x2(i) = v
        K -= 1
      }
      i += 1
    }

    val objNext = objective(x2)

    val relError = (objNext - objCurrent) / objCurrent

    CDState(x2, math.abs(relError), objNext, hessianDiag)

  }
}
