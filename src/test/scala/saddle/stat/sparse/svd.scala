package stat.sparse

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import stat.matops._

class SVDSuite extends FunSuite {

  test("svd 3x2 dense ") {
    val m = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
    // val mfull = m.svd(2)
    val mfull = Svd(m, 2)(DenseMatOps)

    val sigma = Mat(mat.diag(mfull.sigma).T.cols: _*)
    val back = (mfull.u mm sigma mm mfull.vt)
    assert(back.roundTo(10) == m)

    val m1 = Svd(m, 1)(DenseMatOps)

    assert(m1.u.roundTo(7) == Mat(Vec(0.6196295, 0.7848945)))

  }

  test("svd 3x2 sparse ") {
    val m: SMat = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d)).rows.map(r =>
      SVec(Series(r), r.length))
    val mfull = Svd(m, 2)(SparseMatOps)

    val sigma = Mat(mat.diag(mfull.sigma).T.cols: _*)
    val back = (mfull.u mm sigma mm mfull.vt)
    assert(back.roundTo(10) == sparse.dense(m))

    val m1 = Svd(m, 1)(SparseMatOps)

    assert(m1.u.roundTo(7) == Mat(Vec(0.6196295, 0.7848945)))

  }

  test("eigen 2x2") {
    implicit val vo = SparseVecOps
    implicit val vom = SparseMatOps
    // slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
    // slogging.LoggerConfig.level = slogging.LogLevel.DEBUG
    val sm: SMat = Vector(SVec(Series(1 -> 3d, 0 -> 1d), 2),
                          SVec(Series(0 -> 3d, 1 -> 1d), 2))
    val dsm = sparse.dense(sm)

    val eig = (Eigen.eigenDecompositionSymmetric(sm, 2)(SparseMatOps))

    assert(eig.q.roundTo(3) == Mat(Vec(0.707, 0.707), Vec(-0.707, 0.707)))
    assert(eig.lambdaReal.roundTo(2).toVec == Vec(4d, -2d))

  }
}
