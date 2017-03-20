package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class AdaGradStep(point: Vec[Double], acc: Vec[Double]) extends ItState

case class AdaGradUpdater(stepSize: Double) extends Updater[AdaGradStep] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[AdaGradStep]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val past = last.map(_.acc).getOrElse(vec.zeros(j.length))

    val newpast = past + j.map(y => y * y)

    val next = b + (j * newpast.map(g => stepSize / math.sqrt(g + 1E-8)))

    AdaGradStep(next, newpast)

  }
}
