// package stat.sgd
//
// import org.saddle._
// import org.saddle.linalg._
// import org.scalatest.FunSuite
// import stat._
//
// class LMSuite extends FunSuite {
//   test("1") {
//     val x: Vec[Double] = (array.linspace(-1d, 1d): Vec[Double])
//     val y: Vec[Double] = (array.randDouble(50): Vec[Double]) * 0.1 + x * 5 + 5d
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
//     assert(fit.estimatesV.roundTo(0).toVec.toSeq == Vec(5d, 5d).toSeq)
//     // assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
//     // assert(fit.covariate("x1").get._1.slope == 1.34653453394437)
//   }
// }
//
// class LRSuite extends FunSuite {
//   test("random ") {
//     val samples = 1000
//     val columns = 2 * samples
//     val design = mat.randn(samples, columns)
//     val betas: Vec[Double] = vec.randn(20) concat vec.zeros(2 * samples - 20)
//     val ly = (design mm Mat(betas)).col(0)
//     val p = ly.map(l => math.exp(l) / (1 + math.exp(l)))
//     val ri = vec.rand(samples).zipMap(p)((r, p) => if (r < p) 1d else 0d)
//
//     // println(design)
//     // println(ri)
//     // println(betas)
//     // println(ri.sum / ri.length.toDouble)
//
//     val ds = DataSource.fromMat(design, ri, 256, vec.ones(columns), 42)
//     // val fit =
//     //   Sgd.optimize(ds,
//     //                sgd.LogisticRegression,
//     //                L2(1d),
//     //                NewtonUpdater,
//     //                300,
//     //                10,
//     //                10,
//     //                1E-6)
//     //
//     //
//     // println(fit)
//
//     val fitFista = Sgd.optimize(ds,
//                                 sgd.LogisticRegression,
//                                 L1(0.1d),
//                                 FistaUpdater,
//                                 3000,
//                                 2,
//                                 10,
//                                 5E-2)
//
//     println(fitFista)
//     val lypredicted = (design mm Mat(fitFista.estimatesV)).col(0)
//     val pp = lypredicted.map(l => math.exp(l) / (1 + math.exp(l)))
//     val hard = pp.map(p => if (p > 0.5) 1.0 else 0.0)
//     val acc = ri
//         .zipMap(hard)((r, p) => if (r == p) 1.0 else 0.0)
//         .sum / ri.length
//     println(acc)
//     // val rmse = ((ly - lypredicted) dot (ly - lypredicted))
//     // println(rmse)
//
//     // assert(fit.estimatesV.roundTo(0).toVec.toSeq == Vec(0d,0d,0d).toSeq)
//     // assert(fit.covariate("intercept").get._1.slope.roundTo(10) == 5.0)
//     // assert(fit.covariate("x1").get._1.slope == 1.34653453394437)
//   }
// }
