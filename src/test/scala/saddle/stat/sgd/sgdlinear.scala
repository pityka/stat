// package stat.sgd
//
// import org.saddle._
// import org.scalatest.FunSuite
// import stat._
//
// class LMSuite extends FunSuite {
//   test("1") {
//     val x: Vec[Double] = (array.linspace(-1d, 1d): Vec[Double])
//     val y: Vec[Double] = (array.randDouble(50): Vec[Double]) * 1 + x * 5 + 5d
//     val data =
//       Mat(vec.ones(50), x)
//
//     val ds = DataSource.fromMat(data, y, 10, Vec(0d, 1d), 42)
//     val fit =
//       Sgd.optimize(ds,
//                    sgd.LinearRegression,
//                    L2(0d),
//                    NewtonUpdater,
//                    1000,
//                    100,
//                    10,
//                    1E-6)
//     println(fit)
//     // assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
//     // assert(fit.covariate("x1").get._1.slope == 1.34653453394437)
//   }
// }
