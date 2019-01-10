package stat.regression

import org.saddle._
import org.saddle.linalg._
import scala.util.{Try, Success, Failure}
import stat.matops._
case class LinearRegressionResult(
    betas: Vec[Double],
    sds: Vec[Double],
    logLikelihood: LogLikelihood,
    r2: Double,
    sigma: Double,
    df: Double,
    lambda: Double,
    adjR2: Double,
    numberOfSamples: Int,
    externallyStudentizedResiduals : Vec[Double]
)

case class NamedLinearRegressionResult[I](
    parameterNames: Index[String],
    sampleNames: Index[I],
    raw: LinearRegressionResult,
    yName: String
) extends RegressionResult {
  import raw._

  def estimatesV = raw.betas

  def names = parameterNames

  def logLikelihood = raw.logLikelihood

  def lambda = raw.lambda

  def numberOfSamples = raw.numberOfSamples

  def predict(v: Vec[Double]): Double =
    v dot betas

  def predict(m: Mat[Double]): Vec[Double] = m mv betas

  def predict[T: MatOps](m: T): Vec[Double] = m mv betas

  def covariates = parameterNames.toSeq.map(x => x -> covariate(x).get).toMap

  def covariate(s: String) = {
    if (parameterNames.contains(s)) {
      val i = parameterNames.getFirst(s)
      val b = betas.raw(i)
      val sd = sds.raw(i)
      Some((Effect(b, sd), StudentTest(b, sd, numberOfSamples - 2)))
    } else None
  }

  def intercept = covariates("intercept")

  def outlierPValues = {
    val df = numberOfSamples - betas.length  -1 
    externallyStudentizedResiduals.map{ residual =>
      2 *jdistlib.T
            .cumulative(math.abs(residual), df.toDouble, false, false)
    }
  }

}

object LinearRegression {

  def linearRegression[I](
      data: Frame[I, String, Double],
      yKey: String,
      lambda: Double = 0.0,
      addIntercept: Boolean = true
  )(implicit ev: org.saddle.ST[I],
    ord: Ordering[I]): NamedLinearRegressionResult[I] = {

    val data2 =
      createDesignMatrix(data, addIntercept)

    val withoutNames = linearRegression(
      X = data2.filterIx(_ != yKey).toMat,
      y = data2.firstCol(yKey).toVec,
      shrinkage = lambda,
      penalizationWeights =
        data2.filterIx(_ != yKey).stdev.toVec.map(x => 1.0 / x)
    )

    NamedLinearRegressionResult(data2.colIx.toSeq.filter(_ != yKey).toIndex,
                                data2.rowIx,
                                withoutNames,
                                yKey)

  }

  def linearRegression(
      X: Mat[Double],
      y: Vec[Double],
      shrinkage: Double,
      penalizationWeights: Vec[Double]
  ): LinearRegressionResult = {

    val Y = Mat(y)

    val XtX = X.innerM

    val (xtXplusLambdaIInverse, xtXplusLambdaI) =
      if (shrinkage == 0.0) (XtX.invertPD.get,XtX)
      else
        {
          val a  =(XtX + (mat.diag(penalizationWeights) * shrinkage))

          (a.invert,a)
        }

    val XTmultY = X.tmm(Y)

    linearRegressionSecondPart(X,
                               Some(XtX),
                               XTmultY,
                               xtXplusLambdaIInverse,
                               xtXplusLambdaI,
                               y,
                               X.numRows,
                               X.numCols,
                               shrinkage)

  }

  private def linearRegressionSecondPart(X: Mat[Double],
                                         XtX: Option[Mat[Double]],
                                         XTmultY: Mat[Double],
                                         XtXplusLambdaIInverse: Mat[Double],
                                         XtXplusLambdaI: Mat[Double],
                                         y: Vec[Double],
                                         numSamples: Int,
                                         numParameters: Int,
                                         shrinkage: Double) = {

    val estimator: Mat[Double] = XtXplusLambdaIInverse.mm(XTmultY)

    val predicted: Vec[Double] = X.mm(estimator).col(0)

    val error: Vec[Double] = (y - predicted).col(0)

    val RSS = error dot error

    val sigma2 = (1.0 / (numSamples - numParameters)) * RSS

    val variances: Mat[Double] =
      if (shrinkage == 0.0) XtXplusLambdaIInverse * sigma2
      else {
        XtXplusLambdaIInverse.mm(XtX.get).mm(XtXplusLambdaIInverse) * sigma2
      }

    // diagonal of the Hat matrix
    // H = X inv(X'X) X'
    val leverages = XtXplusLambdaI.diagInverseSandwich(X).get

    val df =
      if (shrinkage == 0.0) numParameters + 1.0
      else
        leverages.sum

    val totalSS = {
      val d = y - y.mean
      d dot d
    }

    val rSquared = 1.0 - (RSS / totalSS)

    val betas = estimator.col(0)

    val paramsds = variances.diag.map(math.sqrt)

    val residuals = error.col(0)

    val externallyStudentizedResiduals = residuals.zipMap(leverages){
      case (residual,leverage) =>
      val correctedSigma2 = (1d / (numSamples - numParameters - 1)) * (RSS - residual * residual) 
      residual / (math.sqrt(correctedSigma2 * (1d - leverage)) )
    }

    val adjr2 = {
      val n = residuals.length.toDouble
      val p = (X.numCols - 1).toDouble
      1.0 - (1.0 - rSquared) * ((n - 1) / (n - p - 1))
    }

    // from R source: src/library/stats/R/logLik.R
    // .5* (sum(log(w)) - N * (log(2 * pi) + 1 - log(N) +
    // log(sum(w*res^2))))
    val logLikelihood = {
      import scala.math.log
      import scala.math.Pi

      val residualSqSum = residuals dot residuals

      val l = -0.5 * X.numRows * (log(2 * Pi) + 1 - log(X.numRows) + log(
          residualSqSum))
      LogLikelihood(L = l, df = df, numberOfSamples = X.numRows)
    }

    LinearRegressionResult(
      betas = betas,
      sds = paramsds,
      logLikelihood = logLikelihood,
      r2 = rSquared,
      sigma = math.sqrt(sigma2),
      df = df,
      lambda = shrinkage,
      adjR2 = adjr2,
      numberOfSamples = X.numRows,
      externallyStudentizedResiduals = externallyStudentizedResiduals
    )

  }

}
