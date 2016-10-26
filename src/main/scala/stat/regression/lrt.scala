package stat.regression

case class LogLikelihood(L: Double, df: Double, numberOfSamples: Int)

case object BIC {
  def apply(l: LogLikelihood) = -2.0 * l.L + l.df * math.log(l.numberOfSamples)
}

case class ChiSqTestResult(statistic: Double, pValue: Double, df: Double)
    extends TestResult

object LikelihoodRatioTest {

  def apply(nullModel: Double,
            alternative: Double,
            df: Double): ChiSqTestResult =
    apply(nullModel, alternative, df, 1.0)

  def apply(nullModel: Double,
            alternative: Double,
            df: Double,
            scale: Double): ChiSqTestResult = {

    val stat = 2 * (alternative - nullModel) / scale

    val p =
      jdistlib.ChiSquare.cumulative(stat, df, false, false)
    
    ChiSqTestResult(stat, p, df)
  }

  def apply(nullModel: LogLikelihood,
            alternative: LogLikelihood): ChiSqTestResult = {
    assert(nullModel.numberOfSamples == alternative.numberOfSamples)
    val dfeff = {
      val df = alternative.df - nullModel.df
      if (df < 1 && math.abs(df - 1.0) < 1E-4) 1
      else if (df < 1)
        throw new RuntimeException(
          "df < 1 (" + df.toString + nullModel + alternative + ")")
      else df
    }

    apply(nullModel.L, alternative.L, dfeff)
  }

}
