package stat.kmeans

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._

class KMeansSuite extends FunSuite {
  test("short") {
    val data = mat.randn(10000, 2)
    val res = kmeans.apply(data, mat.randn(18, 2), 10)

    show(plot(data, res, 2))

  }

}
