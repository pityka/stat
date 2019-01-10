package stat.regression

import org.saddle._
import org.scalatest.FunSuite
import stat._

class LMSuite extends FunSuite {
  test("1") {
    val y: Vec[Double] = array.linspace(-1d, 1d)
    val data = Frame(
      Mat(y + 5d, y * y * y),
      Index(0 until 50: _*),
      Index("y", "x1")
    )
    val fit = LinearRegression.linearRegression(data, "y")
    assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
    assert(
      fit.covariate("x1").get._1.slope.roundTo(7) == 1.34653453394437.roundTo(
        7))
  }
    test("2") {
    val y: Vec[Double] = array.linspace(-10d, 10d, 150)
    val data = Frame(
      Mat(y*3 +vec.randn(150), y ),
      Index(0 until 150: _*),
      Index("y", "x1")
    )
    val fit = LinearRegression.linearRegression(data, "y")
    assert(math.abs(stat.vis.QQUniform.computeLambda((fit.outlierPValues.toSeq)) - 1) < 0.15d)
    assert(math.abs(fit.covariate("intercept").get._1.slope) < 0.1)
    assert(
      math.abs(fit.covariate("x1").get._1.slope -3)  < 0.1)
  }
}
