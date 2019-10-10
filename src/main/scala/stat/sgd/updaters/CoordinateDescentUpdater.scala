package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._
import stat.matops._
import slogging.StrictLogging

case class CDState(point: Vec[Double],
                   hessianDiag: Vec[Double],
                   activeSet: Int,
                   work: Array[Double])
    extends ItState {
  override def toString =
    "CDState"
}

case class CoordinateDescentUpdater(doActiveSet: Boolean = false)
    extends Updater[CDState]
    with StrictLogging {
  def next[M: MatOps](x: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[CDState]): CDState = {

    val penalizationMask = obj.adaptPenalizationMask(batch)

    val hessianDiag = last.map(_.hessianDiag).getOrElse {
      val xmvb = batch.x mv x
      0 until x.length map { i =>
        obj.hessian1D(x, batch, i, None, xmvb)
      } toVec
    }

    val work = last.map(_.work).getOrElse {
      Array.ofDim[Double](batch.y.length)
    }

    def objective(w: Vec[Double]): (Double, Double) = {
      val x = objectiveUnpenalized(w)
      (x * (-1), x + penaltyFunction(w))
    }

    def penaltyFunction(w: Vec[Double]): Double =
      pen.apply(w, penalizationMask)

    def shrink(w: Double, alpha1: Double, pm: Double) =
      pen.proximal1D(w, pm, alpha1)

    def objectiveUnpenalized(w: Vec[Double]): Double =
      -1 * obj.apply(w, batch, Some(work))

    /* 1D Newton, then shrink. */
    def updateCoordinate(i: Int, x: Vec[Double], xmvb: Vec[Double]) = {
      val stepSize = {
        val s = (-1d) / obj
            .hessian1D(x, batch, i, Some(hessianDiag.raw(i)), xmvb)
        if (s.isInfinite) 1E-16 else s
      }
      val gradient = obj.jacobi1D(x, batch, i, xmvb) * (-1)
      shrink(x.raw(i) - gradient * stepSize, stepSize, penalizationMask.raw(i))
    }

    val matops = implicitly[MatOps[M]]
    import matops.vops

    val activeSetMode =
      if (!doActiveSet) false
      else
        last.map(x => x.activeSet < 100).getOrElse(false)

    // mutated
    val xmvb: Array[Double] = (batch.x mv x).toArray

    // mutated
    val x2: Array[Double] = {
      val ar = Array.ofDim[Double](x.length)
      System.arraycopy(x.toArray, 0, ar, 0, x.length)
      ar
    }

    val coordinates =
      if (activeSetMode) x.find(_ != 0.0)
      else Vec(0 until x.length: _*)

    // logger.trace("Coordinates: {}", coordinates.length)

    var k = 0
    val N = coordinates.length
    while (k < N) {
      val i = coordinates.raw(k)

      val v = updateCoordinate(i, x2.toVec, xmvb.toVec)
      val oldV = x2(i)
      var j = 0
      val n = xmvb.size
      while (j < n) {
        xmvb(j) += matops.raw(batch.x, j, i) * (v - oldV)
        j += 1
      }

      x2(i) = v
      k += 1
    }

    val activeSetCounter =
      if (!activeSetMode) 0 else last.map(_.activeSet + 1).getOrElse(0)

    CDState(x2.toVec, hessianDiag, activeSetCounter, work)

  }
}
