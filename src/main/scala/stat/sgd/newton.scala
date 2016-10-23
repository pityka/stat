package stat.sgd

import org.saddle._
import org.saddle.linalg._

case class Iteration(point: Vec[Double], lprimesum: Double) extends ItState

object NewtonUpdater extends Updater[Iteration] {
  def next(b: Vec[Double],
           batch: Batch,
           obj: ObjectiveFunction,
           pen: Penalty,
           last: Option[Iteration]) = {

    val j = obj.jacobi(b, batch) - pen.jacobi(b, batch)
    val h = obj.hessian(b, batch) - pen.hessian(b, batch)

    val hinv =
      (h * (-1)).invertPD
        .map(_ * (-1))
        .getOrElse(h.invertPD.getOrElse(h.invert))

    val next = b - (hinv mm j).col(0)

    val jacobisum =
      (obj.jacobi(next, batch) - pen.jacobi(next, batch)).map(math.abs).sum

    Iteration(next, jacobisum)

  }
}
