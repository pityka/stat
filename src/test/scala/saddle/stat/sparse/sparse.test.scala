package stat.sparse

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._

class SparseSuite extends FunSuite {
  test("dense") {
    val sm = Vector(Series(0 -> 2d), Series(2 -> 4d))
    assert(dense(sm) == Mat(Vec(2d, 0d), Vec(0d, 0d), Vec(0d, 4d)))
  }
}
