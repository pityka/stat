package stat.sgd

import org.saddle._
import org.saddle.linalg._

trait Penalty[H] {
  def proximal(b: Vec[Double],
               penalizationMask: Vec[Double],
               alpha: Double): Vec[Double]
  def jacobi(b: Vec[Double], penalizationMask: Vec[Double]): Vec[Double]
  def hessian(p: Vec[Double], penalizationMask: Vec[Double]): Mat[Double]
  def apply(b: Vec[Double], penalizationMask: Vec[Double]): Double
  def withHyperParameter(h: H): Penalty[H]
  def proximal1D(b: Double, penalizationMask: Double, alpha: Double): Double
}

case object NoPenalty extends Penalty[Nothing] {
  def proximal(b: Vec[Double],
               penalizationMask: Vec[Double],
               alpha: Double): Vec[Double] = b
  def jacobi(b: Vec[Double], penalizationMask: Vec[Double]): Vec[Double] =
    vec.zeros(b.length)
  def hessian(p: Vec[Double], penalizationMask: Vec[Double]): Mat[Double] =
    mat.diag(penalizationMask * 0d)
  def apply(b: Vec[Double], penalizationMask: Vec[Double]): Double = 0d
  def withHyperParameter(h: Nothing): Penalty[Nothing] = NoPenalty
  def proximal1D(b: Double, penalizationMask: Double, alpha: Double): Double =
    b
}

case class L2(lambda: Double) extends Penalty[Double] {
  def jacobi(b: Vec[Double], penalizationMask: Vec[Double]): Vec[Double] =
    b * penalizationMask * lambda

  def hessian(p: Vec[Double], penalizationMask: Vec[Double]): Mat[Double] =
    mat.diag(penalizationMask * lambda)

  def proximal(w: Vec[Double],
               penalizationMask: Vec[Double],
               alpha: Double): Vec[Double] =
    w.zipMap(penalizationMask)((old, pw) => proximal1D(old, pw, alpha))

  def proximal1D(b: Double, penalizationMask: Double, alpha: Double): Double =
    b / (1.0 + penalizationMask * lambda * alpha)

  def apply(b: Vec[Double], penalizationMask: Vec[Double]): Double =
    (b.map(x => x * x) * penalizationMask * 0.5 * lambda).sum

  def withHyperParameter(h: Double) = L2(h)
}

case class L1(lambda: Double) extends Penalty[Double] {

  def proximal1D(w: Double, penalizationMask: Double, alpha: Double) =
    math.signum(w) * math.max(0.0,
                              math.abs(w) - lambda * alpha * penalizationMask)

  def proximal(w: Vec[Double], penalizationMask: Vec[Double], alpha: Double) =
    w.zipMap(penalizationMask)((old, pw) => proximal1D(old, pw, alpha))

  def jacobi(b: Vec[Double], penalizationMask: Vec[Double]): Vec[Double] =
    throw new RuntimeException("L1 jacobi")

  def hessian(p: Vec[Double], penalizationMask: Vec[Double]): Mat[Double] =
    throw new RuntimeException("L1 hessian")

  def apply(b: Vec[Double], penalizationMask: Vec[Double]): Double =
    (b.map(math.abs) * penalizationMask * 0.5 * lambda).sum

  def withHyperParameter(h: Double) = L1(h)

}

case class ElasticNet(lambda1: Double, lambda2: Double)
    extends Penalty[(Double, Double)] {

  def proximal1D(old: Double, pw: Double, alpha: Double): Double =
    if (pw.isPosInfinity) 0.0
    else
      math.signum(old) * math.max(
        0.0,
        (math.abs(old) - lambda1 * alpha * pw) / (1.0 + pw * alpha * lambda2))

  def proximal(w: Vec[Double], penalizationMask: Vec[Double], alpha: Double) =
    w.zipMap(penalizationMask)((old, pw) => proximal1D(old, pw, alpha))

  def jacobi(b: Vec[Double], penalizationMask: Vec[Double]): Vec[Double] =
    throw new RuntimeException("EN Jacobi")

  def hessian(p: Vec[Double], penalizationMask: Vec[Double]): Mat[Double] =
    throw new RuntimeException("EN Hessian")

  def apply(b: Vec[Double], penalizationMask: Vec[Double]): Double =
    (b.map(math.abs) * penalizationMask * 0.5 * lambda1).sum +
      (b.map(x => x * x) * penalizationMask * 0.5 * lambda2).sum

  def withHyperParameter(h: (Double, Double)) = ElasticNet(h._1, h._2)

}
