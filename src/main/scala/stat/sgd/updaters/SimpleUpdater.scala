package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class SimpleUpdater(stepSize: Double) extends Updater[Iteration] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[Iteration]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val next = b + (j * stepSize)

    Iteration(next)

  }
}
