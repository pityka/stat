package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._

case class FistaItState(point: Vec[Double],
                        convergence: Double,
                        t: Double,
                        y: Vec[Double],
                        stepSize: Double,
                        obj: Double)
    extends ItState

/**
  * accelerated proximal gradient descent (FISTA)
  *
  * ref: https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf
  */
object FistaUpdater extends Updater[FistaItState] {
  def next(x: Vec[Double],
           batch: Batch,
           obj: ObjectiveFunction[_],
           pen: Penalty,
           last: Option[FistaItState]): FistaItState = {

    def shrink(w: Vec[Double], alpha1: Double) =
      pen.proximal(w, batch, alpha1)

    def penaltyFunction(w: Vec[Double]): Double = pen.apply(w, batch)

    def gradient(w: Vec[Double]) = obj.jacobi(w, batch) * (-1)

    def step(stepSize: Double, w: Vec[Double], gradientw: Vec[Double]) =
      shrink(
        w - (gradientw * stepSize),
        stepSize
      )

    def objective(w: Vec[Double]): Double =
      objectiveUnpenalized(w) + penaltyFunction(w)

    def objectiveUnpenalized(w: Vec[Double]): Double = -1 * obj.apply(w, batch)

    def quad(x: Vec[Double], y: Vec[Double], a: Double) =
      objectiveUnpenalized(y) + (gradient(y) dot (x - y)) + (1.0 / (2.0 * a)) * ((x - y) dot (x - y)) + penaltyFunction(
        x)

    val t = last.map(_.t).getOrElse(1.0)
    val y = last.map(_.y).getOrElse(x)

    /* 1 / Lipschitz constant */
    val stepSize = last.map(_.stepSize).getOrElse {
      val e = obj.minusHessianLargestEigenValue(x, batch) * 2
      1 / e
    }

    val xnext = step(stepSize, y, gradient(y))

    val tplus1 = (1 + math.sqrt(1 + 4 * t * t)) * 0.5

    val ynext = xnext + (xnext - x) * ((t - 1.0) / tplus1)

    val objCurrent = last.map(_.obj).getOrElse(objective(x))

    val objNext = objective(xnext)

    val relError = (objNext - objCurrent) / objCurrent

    FistaItState(xnext, math.abs(relError), tplus1, ynext, stepSize, objNext)

  }
}
