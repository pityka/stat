package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class AdaDeltaStep(point: Vec[Double],
                        acc1: Vec[Double],
                        acc2: Vec[Double])
    extends ItState

object AdaDeltaUpdater extends Updater[AdaDeltaStep] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[AdaDeltaStep]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val past1 = last.map(_.acc1).getOrElse(vec.zeros(j.length))
    val past2 = last.map(_.acc2).getOrElse(vec.zeros(j.length))

    val newpast1 = past1 * 0.95 + j.map(x => x * x * 0.05)

    val update = (j * newpast1.zipMap(past2)((p1, p2) =>
        math.sqrt(p2 + 1E-8) / math.sqrt(p1 + 1E-8)))

    val next = b + update

    val newpast2 = past2 * 0.95 + update.map(x => x * x * 0.05)

    AdaDeltaStep(next, newpast1, newpast2)

  }
}
