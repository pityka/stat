package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._
import stat.matops._
import slogging.StrictLogging

case class FistaItState(point: Vec[Double],
                        t: Double,
                        x: Vec[Double],
                        stepSize: Double,
                        minStepSize: Double)
    extends ItState {
  override def toString =
    s"FistaItState(stepSize=$stepSize)"
}

/**
  * accelerated proximal gradient descent (FISTA)
  *
  * ref: https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf
  */
object FistaUpdater extends Updater[FistaItState] with StrictLogging {
  def next[M: MatOps](y: Vec[Double],
                      batch: Batch[M],
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

    def objective(w: Vec[Double]): (Double, Double) = {
      val x = objectiveUnpenalized(w)
      (x * (-1), x + penaltyFunction(w))
    }

    def objectiveUnpenalized(w: Vec[Double]): Double = -1 * obj.apply(w, batch)

    def quad(x: Vec[Double], y: Vec[Double], a: Double) =
      objectiveUnpenalized(y) + (gradient(y) dot (x - y)) + (1.0 / (2.0 * a)) * ((x - y) dot (x - y)) + penaltyFunction(
        x)

    val t = last.map(_.t).getOrElse(1.0)
    val x = last.map(_.x).getOrElse(y)

    /* 1 / Lipschitz constant */
    val minStepSize = last.map(_.minStepSize).getOrElse {
      val e = obj.minusHessianLargestEigenValue(y, batch) * 2
      1 / e
    }

    val stepSize = last.map(_.stepSize).getOrElse(minStepSize * 1E4)

    val gradY = gradient(y)

    def trystep(stepSize: Double): (Double, Vec[Double], Vec[Double], Double) = {
      val xnext = step(stepSize, y, gradY)

      val tplus1 = (1 + math.sqrt(1 + 4 * t * t)) * 0.5

      val ynext = xnext + (xnext - x) * ((t - 1.0) / tplus1)

      val (_, objNext) = objective(ynext)

      val quadApprox = quad(xnext, y, stepSize)

      if (quadApprox < objNext && stepSize > 2 * minStepSize) {
        logger.trace(
          "Halfing step size Q={} < F={}, current step: {}, min step size reached: {}",
          quadApprox,
          objNext,
          stepSize,
          stepSize < 2 * minStepSize)
        trystep(math.max(stepSize * 0.5, minStepSize))
      } else {
        logger.trace(
          "Keeping step size Q={} >= F={}, min step size reached: {}",
          quadApprox,
          objNext,
          stepSize < 2 * minStepSize)
        (tplus1, xnext, ynext, stepSize)
      }

    }

    val (tplus1, xnext, ynext, step2) = trystep(stepSize)

    FistaItState(ynext, tplus1, xnext, step2, minStepSize)
  }

}
