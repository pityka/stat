package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._
import stat.matops._

case class CDState(point: Vec[Double], convergence: Double, obj: Double)
    extends ItState

object CoordinateDescentUpdater extends Updater[CDState] {
  def next[M: MatOps](x: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[CDState]): CDState = {

    val penalizationMask = obj.adaptPenalizationMask(batch)

    def objective(w: Vec[Double]): Double =
      objectiveUnpenalized(w) + penaltyFunction(w)

    def penaltyFunction(w: Vec[Double]): Double =
      pen.apply(w, penalizationMask)

    def shrink(w: Double, alpha1: Double, pm: Double) =
      pen.proximal1D(w, pm, alpha1)

    def objectiveUnpenalized(w: Vec[Double]): Double = -1 * obj.apply(w, batch)

    /* 1D Newton, then shrink. */
    def updateCoordinate(i: Int, x: Vec[Double]) = {
      val stepSize = (-1d) / obj.hessian1D(x, batch, i)
      val gradient = obj.jacobi1D(x, batch, i) * (-1)

      shrink(x.raw(i) - gradient * stepSize, stepSize, penalizationMask.raw(i))
    }

    val objCurrent = last.map(_.obj).getOrElse(objective(x))

    val xAfterSweep = (0 until x.length).foldLeft(x) {
      case (x, i) =>
        val v = updateCoordinate(i, x)
        val x2 = x.toSeq.updated(i, v).toVec // TODO
        x2
    }

    val objNext = objective(xAfterSweep)

    val relError = (objNext - objCurrent) / objCurrent

    CDState(xAfterSweep, relError, objNext)

  }
}
