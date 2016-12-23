package stat.sparse

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import stat.matops._

class SVDSuite extends FunSuite {

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
