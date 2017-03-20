package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class Iteration(point: Vec[Double]) extends ItState

object NewtonUpdater extends Updater[Iteration] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[Iteration]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)
    val h = obj.hessian(b, batch) - pen.hessian(b, penalizationMask)

    val hinv =
      (h * (-1)).invertPD
        .map(_ * (-1))
        .getOrElse(h.invertPD.getOrElse(h.invert))

    val next = b - (hinv mm j).col(0)

    // val jacobisum =
    //   (obj.jacobi(next, batch) - pen.jacobi(next, penalizationMask))
    //     .map(math.abs)
    //     .sum

    Iteration(next)

  }
}
