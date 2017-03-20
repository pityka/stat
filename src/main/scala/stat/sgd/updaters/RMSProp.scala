package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class RMSPropStep(point: Vec[Double], acc: Vec[Double]) extends ItState

case class RMSPropUpdater(stepSize: Double) extends Updater[RMSPropStep] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[RMSPropStep]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val past = last.map(_.acc).getOrElse(vec.zeros(j.length))

    val newpast = past * 0.95 + j.map(x => x * x * 0.05)

    val next = b + (j * newpast.map(g => stepSize / math.sqrt(g + 1E-8)))

    RMSPropStep(next, newpast)

  }
}
