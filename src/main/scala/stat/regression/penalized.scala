package stat.regression

import org.saddle._
import org.saddle.linalg._
import scala.util._

sealed trait PenaltyType
case object L1 extends PenaltyType
case object L2 extends PenaltyType
case object SCAD extends PenaltyType
case object ElasticNet extends PenaltyType

case class PenalizedLeastSquaresResult(estimatesV: Vec[Double],
                                       lambda1: Double,
                                       lambda2: Double,
                                       objectiveFunction: Double,
                                       iterations: Int,
                                       penalty: PenaltyType)

case class NamedPenalizedLeastSquaresResult[L](
    estimates: Series[String, Double],
    lambda: L)
    extends Prediction {
  def predict(v: Vec[Double]) = v dot estimatesV
}

trait PenalizedRegression[L] {
  def doFit(
      design: Mat[Double],
      y: Vec[Double],
      lambda: L,
      penalizationWeights: Vec[Double],
      first: Option[Vec[Double]]
  ): PenalizedLeastSquaresResult

  def fit[I: ST: ORD](
      data: Frame[I, String, Double],
      yKey: String,
      lambda: L,
      missingMode: MissingMode = DropSample,
      unpenalized: Seq[String] = Nil,
      addIntercept: Boolean = true,
      standardize: Boolean = false
  ): NamedPenalizedLeastSquaresResult[L] = {

    val data2 =
      createDesignMatrix(data, missingMode, addIntercept)

    val penalizationWeights = data2.toColSeq
      .filterNot(_._1 == yKey)
      .map {
        case (cx, series) =>
          if (cx == "intercept" || unpenalized.contains(cx)) 0.0
          else (if (standardize) 1.0 / series.toVec.stdev
                else 1.0)
      }
      .toVec

    val result = doFit(data2.filterIx(_ != yKey).toMat,
                       data2.firstCol(yKey).toVec,
                       lambda,
                       penalizationWeights,
                       None)

    NamedPenalizedLeastSquaresResult(
      Series(result.estimatesV,
             Index(data2.colIx.toSeq.filterNot(_ == yKey): _*)),
      lambda)

  }
}

object Ridge extends PenalizedRegression[Double] {
  def doFit(
      design: Mat[Double],
      y: Vec[Double],
      lambda: Double,
      penalizationWeights: Vec[Double],
      first: Option[Vec[Double]]
  ) =
    LASSOImpl.fista(design,
                    y,
                    Double.NaN,
                    lambda,
                    penalizationWeights,
                    L2,
                    first = first)
}
//
// object RidgeClosedForm extends PenalizedRegression[LinearRegressionResult, Double] {
//   def fit(
//     design: Mat[Double],
//     y: Vec[Double],
//     lambda: Double,
//     penalizationWeights: Vec[Double],
//     first: Option[Vec[Double]] = None
//   ) = linearRegression(design, y, lambda, penalizationWeights).toOption
// }
//
object LASSO extends PenalizedRegression[Double] {
  def doFit(
      design: Mat[Double],
      y: Vec[Double],
      lambda: Double,
      penalizationWeights: Vec[Double],
      first: Option[Vec[Double]]
  ) =
    LASSOImpl.fista(design,
                    y,
                    lambda,
                    Double.NaN,
                    penalizationWeights,
                    L1,
                    first = first)

}

object PenalizedWithElasticNet extends PenalizedRegression[(Double, Double)] {
  def doFit(
      design: Mat[Double],
      y: Vec[Double],
      lambda: (Double, Double),
      penalizationWeights: Vec[Double],
      first: Option[Vec[Double]]
  ) =
    LASSOImpl.fista(design,
                    y,
                    lambda._1,
                    lambda._2,
                    penalizationWeights,
                    ElasticNet,
                    first = first,
                    maxIter = 500)
}

object PenalizedWithSCAD extends PenalizedRegression[Double] {
  def doFit(
      design: Mat[Double],
      y: Vec[Double],
      lambda: Double,
      penalizationWeights: Vec[Double],
      first: Option[Vec[Double]]
  ) =
    LASSOImpl.fista(design,
                    y,
                    lambda,
                    Double.NaN,
                    penalizationWeights,
                    SCAD,
                    first = first)
}

object LASSOImpl {

  def lasso(
      design: Mat[Double],
      y: Vec[Double],
      lambda: Double,
      penalizationWeights: Vec[Double],
      relativeThreshold: Double = 1E-4,
      maxIter: Int
  ) =
    fista(design,
          y,
          lambda,
          Double.NaN,
          penalizationWeights,
          L1,
          relativeThreshold,
          maxIter)

  /**
    * accelerated proximal gradient descent (FISTA)
    *
    * ref: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf page 152
    * ref: https://blogs.princeton.edu/imabandit/2013/04/11/orf523-ista-and-fista/
    * ref: http://mechroom.technion.ac.il/~becka/papers/71654.pdf eq 1.4
    * ref for SCAD: http://sites.stat.psu.edu/~rli/research/penlike.pdf
    * ref for SCAD: https://hal.archives-ouvertes.fr/hal-01102810/document !!NOTE! that the scad proximity operator, first case on top of page 4 is wrong there is a typo, absolute value is missing in the divisor. correct is sign(z)*max(0,(|x|-lambda)) OR x*max(0,(1-lambda/|x|))
    */
  def fista(
      design: Mat[Double],
      y: Vec[Double],
      lambda1: Double,
      lambda2: Double,
      penalizationWeights: Vec[Double],
      penalty: PenaltyType,
      relativeThreshold: Double = 1E-6,
      absoluteThreshold: Double = 1E-10,
      maxIter: Int = 1500,
      first: Option[Vec[Double]] = None
  ) = {

    val designT = design.T

    def shrinkElasticNet(w: Vec[Double], alpha1: Double, alpha2: Double) =
      w.zipMap(penalizationWeights)(
        (old, pw) =>
          if (pw.isPosInfinity) 0.0
          else
            math.signum(old) * math
              .max(0.0, (math.abs(old) - alpha1 * pw) / (1.0 + pw * alpha2)))

    def shrinkL1(w: Vec[Double], alpha: Double) =
      w.zipMap(penalizationWeights)((old, pw) =>
        math.signum(old) * math.max(0.0, math.abs(old) - alpha * pw))

    def shrinkL2(w: Vec[Double], alpha: Double) =
      w.zipMap(penalizationWeights)((old, pw) => old / (1.0 + pw * alpha))

    def shrinkSCAD(w: Vec[Double], alpha: Double) =
      w.zipMap(penalizationWeights) { (old, pw) =>
        val pwa = pw * alpha
        val a = 3.7
        if (math.abs(old) <= 2 * pwa)
          math.signum(old) * math
            .max(0.0, (math.abs(old) - pwa)) //old * math.max(0.0, 1.0 - (pwa / math.abs(old)))
        else if (2 * pwa < math.abs(old) && math.abs(old) <= a * pwa)
          (old / (a - 2.0)) * (a - 1 - ((a * pwa) / math.abs(old)))
        else old
      }

    def shrink(w: Vec[Double], alpha1: Double, alpha2: Double) =
      penalty match {
        case L1 => shrinkL1(w, alpha1)
        case L2 => shrinkL2(w, alpha2)
        case SCAD => shrinkSCAD(w, alpha1)
        case ElasticNet => shrinkElasticNet(w, alpha1, alpha2)
      }

    def penaltyFunction(w: Vec[Double]) = penalty match {
      case L1 =>
        lambda1 * (w
          .zipMap(penalizationWeights)((w, pw) => math.abs(w) * pw)
          .sum)
      case L2 =>
        lambda2 * (w.zipMap(penalizationWeights)((w, pw) => w * w * pw).sum)
      case ElasticNet =>
        (w.zipMap(penalizationWeights)((w, pw) =>
            lambda2 * w * w * pw + lambda1 * math.abs(w) * pw)
          .sum)
      case SCAD =>
        (w.zipMap(penalizationWeights) { (w1, pw) =>
            val w = math.abs(w1)
            val a = 3.7
            val pwa = pw * lambda1
            if (w <= pwa) pwa * w
            else if (pwa < w && w <= a * pwa)
              (a * pwa * w - w * w * 0.5) / (a - 1)
            else a * pwa
          })
          .sum
    }

    def gradientAt(w: Vec[Double], residualw: Vec[Double]): Vec[Double] =
      (designT mm Mat(residualw)).col(0)

    def step(stepSize: Double, w: Vec[Double], gradientw: Vec[Double]) = {
      shrink(
        w - (gradientw * stepSize),
        stepSize * lambda1,
        stepSize * lambda2
      )
    }

    def residual(w: Vec[Double]): Vec[Double] = (predict(w) - y)

    def predict(w: Vec[Double]): Vec[Double] = (design mm Mat(w)).col(0)

    def objective(w: Vec[Double]): Double =
      objectiveUnpenalized(w)._1 + penaltyFunction(w)

    def objectiveUnpenalized(w: Vec[Double]): (Double, Vec[Double]) = {
      val r: Vec[Double] = residual(w)
      ((r dot r) * 0.5, r)
    }

    def upperBound(x: Vec[Double],
                   y: Vec[Double],
                   a: Double,
                   objectiveUnpenalizedy: Double,
                   gradienty: Vec[Double]) = {
      objectiveUnpenalizedy + (gradienty dot (x - y)) + (1.0 / (2.0 * a)) * ((x - y) dot (x - y))
    }

    def stop(n1: Double, n2: Double, i: Int, w: Vec[Double]): Boolean = {
      ((n1 > n2 && (n1 - n2) / n2 < relativeThreshold) || (math.abs(n1 - n2) < absoluteThreshold) || i >= maxIter || w.contents
        .forall(_ == 0d))
    }

    def lineSearch(stepSize: Double, w: Vec[Double]): (Double, Vec[Double]) = {
      var l = stepSize

      val (objectiveUnpenalizedw, residualw) = objectiveUnpenalized(w)
      val penaltyw = penaltyFunction(w)
      val objectivePenalizedW = objectiveUnpenalizedw + penaltyw
      val gradientw = gradientAt(w, residualw)

      var z = step(l, w, gradientw)

      def lineSearchStop(z: Vec[Double]) = penalty match {
        case L1 =>
          (objectiveUnpenalized(z)._1 > upperBound(z,
                                                   w,
                                                   l,
                                                   objectiveUnpenalizedw,
                                                   gradientw))
        case SCAD | L2 | ElasticNet => objective(z) > objectivePenalizedW
      }

      while (lineSearchStop(z)) {
        l = l * 0.5
        z = step(l, w, gradientw)
      }

      (l, z)
    }

    var coeff = first.getOrElse(vec.zeros(design.numCols))
    var yCoeff = coeff
    var t = 1.0
    var lastObj = objective(coeff)
    val firstObj = lastObj
    var stopV = false
    var i = 0
    var stepSize = 1.0

    var stream: List[Double] = lastObj :: Nil

    while (!stopV) {

      val (nstepSize, ncoeff) = lineSearch(stepSize, yCoeff)

      stepSize = nstepSize
      val tplus1 = (1 + math.sqrt(1 + 4 * t * t)) * 0.5
      yCoeff = ncoeff + (ncoeff - coeff) * ((t - 1.0) / tplus1)
      coeff = ncoeff
      yCoeff = ncoeff
      t = tplus1

      val newobj = objective(yCoeff)
      i += 1
      stopV = stop(lastObj, newobj, i, yCoeff)
      lastObj = newobj
      stream = lastObj :: stream
    }

    if (i == maxIter) {
      // println(s"WARNING FISTA REACHED MAX ITER $maxIter lambda1=" + lambda1 + " lambda2=" + lambda2 + " step=" + stepSize + " samples=" + design.numRows + " features=" + design.numCols + " lastObj=" + lastObj + " firstObj=" + firstObj + " iterations=" + (stream.take(10) ++ stream.takeRight(10)))
    }

    // println(i)

    PenalizedLeastSquaresResult(yCoeff, lambda1, lambda2, lastObj, i, penalty)

  }

}
