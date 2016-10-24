package stat.sgd

import org.saddle._
import org.saddle.linalg._
import scala.util._

case class FistaItState(point: Vec[Double],
                        convergence: Double,
                        t: Double,
                        y: Vec[Double])
    extends ItState

/**
  * accelerated proximal gradient descent (FISTA)
  *
  * ref: https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf
  */
object FistaUpdater extends Updater[FistaItState] {
  def next(x: Vec[Double],
           batch: Batch,
           obj: ObjectiveFunction,
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

    // def lineSearch(stepSize: Double, w: Vec[Double]): (Double, Vec[Double]) = {
    //   var l = stepSize
    //
    //   val gradientw = gradient(w)
    //
    //   var z = step(l, w, gradientw)
    //
    //   val objw = objective(w)
    //
    //   def lineSearchStop(z: Vec[Double]) = objective(z) > quad(z, w, l)
    //   //   penalty match {
    //   //   case L1 =>
    //   //     (objectiveUnpenalized(z)._1 > upperBound(z,
    //   //                                              w,
    //   //                                              l,
    //   //                                              objectiveUnpenalizedw,
    //   //                                              gradientw))
    //   //   case SCAD | L2 | ElasticNet => objective(z) > objectivePenalizedW
    //   // }
    //
    //   while (lineSearchStop(z)) {
    //     l = l * 0.5
    //     z = step(l, w, gradientw)
    //   }
    //
    //   (l, z)
    // }

    val t = last.map(_.t).getOrElse(1.0)
    val y = last.map(_.y).getOrElse(x)

    // val stepSize = last.map(_.stepSize).getOrElse(1.0)

    // val (nstepSize, xnext) = lineSearch(stepSize, y)

    /* 1 / Lipschitz constant */
    val stepSize = {
      val e = obj.minusHessianLargestEigenValue(x, batch) * 2
      1 / e
    }

    val xnext = step(stepSize, y, gradient(y))

    val tplus1 = (1 + math.sqrt(1 + 4 * t * t)) * 0.5

    val ynext = xnext + (xnext - x) * ((t - 1.0) / tplus1)

    // val jacobisum =
    // (obj.jacobi(xnext, batch) - pen.jacobi(xnext, batch)).map(math.abs).sum

    /* Empirical gradient to test convergence */
    val absError = math.abs(objective(xnext) - objective(x))

    FistaItState(xnext, absError, tplus1, ynext)

  }
}
