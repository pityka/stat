package stat.regression

import stat._
import org.saddle._
import org.saddle.linalg._
import scala.util.{Try, Success, Failure}

sealed trait MissingMode
case object DropSample extends MissingMode
case object MeanImpute extends MissingMode

case class Effect(slope: Double, sd: Double) {
  def slopeOverSD = slope / sd
}

trait TestResult {
  def statistic: Double
  def pValue: Double
}

object TestResult {
  def unapply(t: TestResult) = Some((t.statistic, t.pValue))
}

case class ZTestResult(statistic: Double, pValue: Double) extends TestResult
object ZTest {
  def apply(e: Double, s: Double) =
    ZTestResult(
      math.abs(e / s),
      jdistlib.Normal.cumulative(math.abs(e / s), 0.0, 1.0, false, false) * 2)
}

sealed trait StudentTestResult extends TestResult {
  def statistic: Double
  def pValue: Double
}
case object FailedStudentTest extends StudentTestResult {
  val statistic = Double.NaN
  val pValue = Double.NaN
}
case class SuccessfulStudentTest(statistic: Double, pValue: Double)
    extends StudentTestResult

object StudentTest {
  def apply(estimate: Double,
            sd: Double,
            df: Double,
            location: Double = 0.0): StudentTestResult =
    if (df <= 0 || sd.isNaN || estimate.isNaN || sd.isInfinite || sd == 0.0 || sd > 1E7)
      FailedStudentTest
    else {
      val statistic = (estimate - location) / sd
      SuccessfulStudentTest(
        statistic = statistic,
        pValue = 2 * (jdistlib.T
            .cumulative(math.abs(statistic), df.toDouble, false, false))
      )
    }

}

trait Prediction {
  def estimatesV: Vec[Double]
  def predict(v: Vec[Double]): Double
  def predict(m: Mat[Double]): Vec[Double]
}

trait NamedPrediction extends Prediction {
  def names: Index[String]
  def estimates: Series[String, Double] = Series(estimatesV, names)
  def predict[I: ST: Ordering](m: Frame[I, String, Double],
                               intercept: Boolean): Series[I, Double] = {

    val m2 = (if (intercept) addIntercept(m) else m).reindexCol(names)
    Series(predict(m2.toMat), m2.rowIx)
  }
}

trait RegressionResult extends NamedPrediction with Prediction {
  def covariate(s: String): Option[(Effect, TestResult)]
  def covariates: Map[String, (Effect, TestResult)]
  def intercept: (Effect, TestResult)
  def numberOfSamples: Int
  def yName: String

  def toLine =
    s"""${covariates
      .map(x => x._1 + ":" + x._2._2.statistic + "/" + x._2._2.pValue)
      .mkString(" ")} $numberOfSamples"""

  private def tableLine(d: (String, (Effect, TestResult))) =
    s"${d._1.padTo(12, " ").mkString}\t${d._2._1.slope.roundTo(3).toString.take(12).padTo(12, " ").mkString}\t${d._2._1.sd
      .roundTo(3)
      .toString
      .take(12)
      .padTo(12, " ")
      .mkString}\t${d._2._2.statistic.roundTo(3).toString.take(12).padTo(12, " ").mkString}\t${d._2._2.pValue.toString.padTo(12, " ").mkString}"

  def table = {
    s"""
  |Call: $yName ~  ${covariates.map(_._1).mkString(" + ")}
  |Coefficients:
  |\t\tEstimate    \tStd.Error   \tStatistic   \tp-value
  |${covariates.toSeq.map(x => tableLine(x)).mkString("\n")}
  |N=$numberOfSamples""".stripMargin
  }

  def lambda: Double

  def logLikelihood: LogLikelihood

}
