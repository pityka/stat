package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._

case class FistaItState(point: Vec[Double],
                        convergence: Double,
                        t: Double,
                        y: Vec[Double],
                        stepSize: Double,
                        obj: Double,
                        minStepSize: Double)
    extends ItState

/**
  * accelerated proximal gradient descent (FISTA)
  *
  * ref: https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf
  */
object FistaUpdater extends Updater[FistaItState] {
  def next(x: Vec[Double],
           batch: Batch,
           obj: ObjectiveFunction[_, _],
           pen: Penalty[_],
           last: Option[FistaItState]): FistaItState = {

    val penalizationMask = obj.adaptPenalizationMask(batch)

    def shrink(w: Vec[Double], alpha1: Double) =
      pen.proximal(w, penalizationMask, alpha1)

    def penaltyFunction(w: Vec[Double]): Double =
      pen.apply(w, penalizationMask)

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
    val objCurrent = last.map(_.obj).getOrElse(objective(x))

    /* 1 / Lipschitz constant */
    val minStepSize = last.map(_.minStepSize).getOrElse {
      val e = obj.minusHessianLargestEigenValue(x, batch) * 2
      1 / e
    }

    val stepSize = last.map(_.stepSize).getOrElse(minStepSize * 512)

    val xnext = step(stepSize, y, gradient(y))

    val tplus1 = (1 + math.sqrt(1 + 4 * t * t)) * 0.5

    val ynext = xnext + (xnext - x) * ((t - 1.0) / tplus1)

    val objNext = objective(xnext)

    val relError = (objNext - objCurrent) / objCurrent

    if (objCurrent < objNext && stepSize != minStepSize)
      FistaItState(x,
                   math.abs(relError),
                   t,
                   y,
                   math.max(stepSize * 0.5, minStepSize),
                   objCurrent,
                   minStepSize)
    else
      FistaItState(xnext,
                   math.abs(relError),
                   tplus1,
                   ynext,
                   stepSize,
                   objNext,
                   minStepSize)

  }
}
