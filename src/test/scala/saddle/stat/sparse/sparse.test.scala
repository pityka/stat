package stat.sparse

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import stat.matops._

class SparseSuite extends FunSuite {
  test("dense") {
    val sm = Vector(SVec(Series(0 -> 2d), 3), SVec(Series(2 -> 4d), 3))
    assert(dense(sm) == Mat(Vec(2d, 0d), Vec(0d, 0d), Vec(0d, 4d)))

    // 2 0 0
    // 0 0 4
  }

  test("ops") {
    implicit val vo = SparseVecOps
    implicit val vom = SparseMatOps
    val sm: SMat = Vector(SVec(Series(0 -> 2d), 3), SVec(Series(2 -> 4d), 3))
    val sv = sm.head
    assert(sv.length == 3)
    assert(sv.vv(Vec(0d, 1d, 2d)) == 0d)
    assert(sv.vv(Vec(2d, 1d, 2d)) == 4d)
    assert(sm.mv(Vec(0d, 1d, 2d)) == Vec(0d, 8d))
    assert(sm.mv(Vec(2d, 1d, 2d)) == Vec(4d, 8d))

    assert(sm.tmv(Vec(2d, 1d)) == dense(sm).T.mv(Vec(2d, 1d)))

    assert(sm.innerM == dense(sm).innerM)

    assert(sm.singularValues(1) == dense(sm).singularValues(1))

    assert(
      dense(sm.mDiagFromLeft(Vec(2d, 3d))) == dense(sm).mDiagFromLeft(
        Vec(2d, 3d)))

    assert(sm.tmm(sm) == sm.innerM)

    assert(sm.mm(dense(sm).T) == dense(sm).mm(dense(sm).T))
  }
}
