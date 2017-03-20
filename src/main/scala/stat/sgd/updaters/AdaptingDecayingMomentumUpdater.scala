package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class AdmStep(point: Vec[Double],
                   acc: Vec[Double],
                   decay: Double,
                   momentum: Vec[Double])
    extends ItState

case class AdaptiveDecayingMomentumUpdater(stepSize: Double,
                                           decayRate: Double = 0.9,
                                           momentumRate: Double = 0.909)
    extends Updater[AdmStep] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[AdmStep]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val decay = last.map(_.decay).getOrElse(1.0) * decayRate

    val momentum = last.map(_.momentum).getOrElse(j)

    val past = last.map(_.acc).getOrElse(vec.zeros(j.length))

    val newpast = past + j.map(y => y * y * decay)

    val j2 = momentum * momentumRate + j * (1 - momentumRate)

    val next = b + (j2 * newpast.map(g => stepSize / math.sqrt(g + 1E-8)))

    AdmStep(next, newpast, decay, j2)

  }
}
