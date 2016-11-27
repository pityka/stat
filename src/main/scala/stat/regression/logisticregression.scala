package stat.regression

import org.saddle._
import org.saddle.linalg._
import stat._
import scala.util.{Try, Success, Failure}
import stat.matops._
case class LogisticRegressionResult(betas: Vec[Double],
                                    sds: Vec[Double],
                                    logLikelihood: LogLikelihood,
                                    numberOfSamples: Int)

case class NamedLogisticRegressionResult[I](
    parameterNames: Index[String],
    raw: LogisticRegressionResult,
    yName: String
) extends RegressionResult {
  import raw._

  def estimatesV = raw.betas

  def names = parameterNames

  def logLikelihood = raw.logLikelihood

  def numberOfSamples = raw.numberOfSamples

  val lambda = 0d

  def predict(v: Vec[Double]): Double = {
    val e = math.exp(v dot betas)
    e / (1 + e)
  }

  def predict(m: Mat[Double]): Vec[Double] =
    (m mv betas).map(x => math.exp(x)).map(x => x / (1 + x))

  def predict[T: MatOps](m: T): Vec[Double] =
    (m mv betas).map(x => math.exp(x)).map(x => x / (1 + x))

  def covariates = parameterNames.toSeq.map(x => x -> covariate(x).get).toMap

  def covariate(s: String) = {
    if (parameterNames.contains(s)) {
      val i = parameterNames.getFirst(s)
      val b = betas.raw(i)
      val sd = sds.raw(i)
      Some((Effect(b, sd), ZTest(b, sd)))
    } else None
  }

  def intercept = covariates("intercept")

}

object LogisticRegression {
  def logisticRegression(
      designMatrix: Mat[Double],
      outcome: Vec[Double]
  ): LogisticRegressionResult = {

    sealed trait IterationProto
    case class IterationFailed(e: Exception) extends IterationProto
    case class Iteration(point: Vec[Double], lprimesum: Double)
        extends IterationProto

    assert(outcome.length == designMatrix.numRows)
    assert(designMatrix.numRows > 0 && designMatrix.numCols > 0,
           "designmatrix is empty")

    val numSamples = outcome.length
    val X = designMatrix

    def estimate(maxIter: Int, epsilon: Double) = {

      iteration(maxIter, epsilon).map { last =>
        last match {
          case IterationFailed(e) => throw e
          case Iteration(estimates, obj) => {

            // -1*Hessian, equal to Fisher's information
            val minusSecondDerivative = {
              val estimatedOdds = odds(estimates)
              val w: Vec[Double] = estimatedOdds * (estimatedOdds * (-1) + 1.0)

              calculateXTwX(X, w)
            }

            val negativeDefinite = minusSecondDerivative.isPositiveDefinite

            if (!negativeDefinite)
              throw new RuntimeException("Hessian not negative definite")
            else {
              val inv = minusSecondDerivative.invertPD.get

              val stderrs = inv.diag.map(math.sqrt)

              val logLikelihood = {
                var i = 0
                var l = 0.0
                while (i < outcome.length) {
                  val lc = linCombWithDesign(estimates, i)
                  l += (outcome.raw(i) * lc) - math.log(1 + math.exp(lc))
                  i += 1
                }

                LogLikelihood(L = l, df = estimates.length, outcome.length)
              }

              LogisticRegressionResult(estimates,
                                       stderrs,
                                       logLikelihood,
                                       outcome.length)

            }
          }
        }

      }.getOrElse(throw new RuntimeException("Did not converge"))

    }

    def iteration(max: Int, epsilon: Double): Option[IterationProto] =
      from(vec.zeros(X.numCols))
        .take(max)
        .takeWhile(_ match {
          case IterationFailed(e) => false
          case x: Iteration => x.lprimesum <= epsilon
        })
        .lastOption

    def from(start: Vec[Double]): Stream[IterationProto] = {

      assert(
        start.length == designMatrix.numCols,
        "initial estimate's dimension is wrong, " + start.length + " " + designMatrix.numCols)

      def loop(start: IterationProto): Stream[IterationProto] = start match {
        case IterationFailed(_) => start #:: Stream.Empty
        case x: Iteration => start #:: loop(nextGuess(x.point))
      }

      val f1 = nextGuess(start)
      loop(f1)
    }

    def nextGuess(
        currentGuess: Vec[Double]
    ): IterationProto =
      try {
        // In the paper this is called PI. Odds at the current estimate of betas.
        val currentOdds = odds(currentGuess)

        // n_i * PI_i , the expected number of positive outcomes at the current guess
        // in the paper this is mu
        val expected: Vec[Double] = currentOdds

        // y - expected
        // outComeCounts - expected
        val diff: Vec[Double] = outcome - expected

        // This w is the diagonal of a diagonalmatrix
        val w: Vec[Double] = currentOdds * (currentOdds * (-1) + 1.0)

        // first derivative of the log likelihood function, evaluated at the current guess
        // Xt * diff
        val lprime: Mat[Double] = X tmm Mat(diff)

        // second derivative of the log likelihood at the current guess.
        // this is almost the covariance matrix of the estimates.
        //  XT * w * X
        val minusldoubleprime: Mat[Double] = calculateXTwX(X, w)

        val nextGuess = currentGuess + (minusldoubleprime.invertPD.get mm lprime)
            .col(0)

        val lprimesum =
          lprime.col(0).foldLeft(0.0)((v, t) => t + math.abs(v))

        Iteration(nextGuess, lprimesum)

      } catch {
        case e: Exception => IterationFailed(e)
      }

    def calculateXTwX(X: Mat[Double], diagW: Vec[Double]) =
      X tmm Mat(X.cols.zip(diagW.toSeq).map { case (col, w) => col * w }: _*)

    def linCombWithDesign(currentGuess: Vec[Double], i: Int): Double =
      currentGuess dot designMatrix.row(i)

    def odds(currentGuess: Vec[Double]): Vec[Double] =
      (Mat(currentGuess) tmmt designMatrix).row(0).map { v =>
        val e = math.exp(v)
        e / (1d + e)
      }

    estimate(100, 1E-6)

  }

  def logisticRegression[I: ST: ORD](
      data: Frame[I, String, Double],
      yKey: String,
      missingMode: MissingMode = DropSample,
      addIntercept: Boolean = true
  ): NamedLogisticRegressionResult[I] = {

    val data2 = createDesignMatrix(data, missingMode, addIntercept)

    val raw = logisticRegression(
      data2.filterIx(_ != yKey).toMat,
      data2.firstCol(yKey).toVec.map(v => if (v > 0.0) 1.0 else 0.0)
    )
    NamedLogisticRegressionResult(data2.colIx.toSeq.filter(_ != yKey).toIndex,
                                  raw,
                                  yKey)

  }

  // private def prepareGroupings(includedCovariatesMat: Mat[Double]) = {
  //
  //   // This saves the hashcode and prevents boxing of doubles.
  //   case class MemoizedHashCode(v: Vec[Double]) {
  //     override val hashCode = {
  //       var s = 1
  //       var i = 0
  //       val n = v.length
  //       while (i < n) {
  //         s = 31 * s + com.google.common.primitives.Doubles.hashCode(v.raw(i))
  //         i += 1
  //       }
  //       s
  //     }
  //   }
  //
  //   import scala.collection.mutable.ListBuffer
  //
  //   def groupFrameByRow3(in: Mat[Double]): Seq[Set[Int]] = {
  //     val mmap = collection.mutable
  //       .AnyRefMap[MemoizedHashCode, collection.mutable.Set[Int]]()
  //     val n = in.numRows
  //     (0 until in.numRows).foreach { idx =>
  //       val row = MemoizedHashCode(in.row(idx)) //.contents.toList
  //       mmap.get(row) match {
  //         case None => {
  //           val bs = collection.mutable.Set[Int](idx)
  //           mmap.update(row, bs)
  //         }
  //         case Some(x) => {
  //           x += idx
  //         }
  //       }
  //     }
  //     mmap.map(x => x._2.toSet).toSeq
  //   }
  //
  //   groupFrameByRow3(includedCovariatesMat).map(_.toIndexedSeq)
  // }

  // def logisticRegression(
  //     X: Mat[Double],
  //     y: Vec[Boolean]
  // ): LogisticRegressionResult = {
  //
  //   val groups = prepareGroupings(X)
  //
  //   logisticRegressionWithGrouping(X, y, groups)
  //
  // }
  //
  // private def logisticRegressionWithGrouping[I](
  //     includedCovariatesMat: Mat[Double],
  //     filteredOutcomes: Vec[Boolean],
  // ): LogisticRegressionResult = {
  //
  //   val designMatrix: Mat[Double] =
  //     includedCovariatesMat.row(groups.map(_.head): _*)
  //
  //   val aggregatedBinLengths: Vec[Int] = Vec(groups.map(x => x.size): _*)
  //
  //   val outComeCounts: Vec[Int] = Vec(groups.map(group =>
  //     filteredOutcomes(group: _*).filter(_ == true).length): _*)
  //
  //   if (designMatrix.numCols == 0 && designMatrix.numRows == 0)
  //     throw new RuntimeException("Design matrix is empty")
  //   else {
  //
  //     logisticRegression(designMatrix, outComeCounts, aggregatedBinLengths)
  //
  //   }
  // }

}
