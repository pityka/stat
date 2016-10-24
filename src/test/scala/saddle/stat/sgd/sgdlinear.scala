package stat.sgd

import org.saddle._
import org.scalatest.FunSuite
import stat._

class LMSuite extends FunSuite {
  test("1") {
    val x: Vec[Double] = (array.linspace(-1d, 1d): Vec[Double])
    val y: Vec[Double] = (array.randDouble(50): Vec[Double]) * 0.1 + x * 5 + 5d
    val data =
      Mat(vec.ones(50), x)

    val ds = DataSource.fromMat(data, y, 10, Vec(0d, 1d), 42)
    val fit =
      Sgd.optimize(ds,
                   sgd.LinearRegression,
                   L2(0d),
                   NewtonUpdater,
                   1000,
                   100,
                   10,
                   1E-6)
    assert(fit.estimates.roundTo(0).toVec.toSeq == Vec(5d, 5d).toSeq)
    // assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
    // assert(fit.covariate("x1").get._1.slope == 1.34653453394437)
  }
}

class LRSuite extends FunSuite {
  test("random ") {
    val x: Vec[Double] = array.randDouble(50)
    val x2: Vec[Double] = array.randDouble(50)
    val x3: Vec[Double] = array.randDouble(50)
    val y = x.map(x => if (x > 0) 1.0 else 0.0)

    val data =
      Mat(vec.ones(50), x, x2)

    val ds = DataSource.fromMat(data, y, 50, Vec(0d, 1d, 1d), 42)
    val fit =
      Sgd.optimize(ds,
                   sgd.LogisticRegression,
                   L2(1d),
                   NewtonUpdater,
                   300,
                   10,
                   10,
                   1E-6)

    println(fit)

    val fitFista = Sgd.optimize(ds,
                                sgd.LogisticRegression,
                                L1(10d),
                                FistaUpdater,
                                300,
                                10,
                                10,
                                1E-6)

    println(fitFista)

    // assert(fit.estimates.roundTo(0).toVec.toSeq == Vec(0d,0d,0d).toSeq)
    // assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
    // assert(fit.covariate("x1").get._1.slope == 1.34653453394437)
  }
}
