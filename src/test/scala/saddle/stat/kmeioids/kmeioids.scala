package stat.kmedoids

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._

class KMedoidsSuite extends FunSuite {
  test("short") {
    val data = mat.randn(20000, 2)
    val dist = data.outerM

    val res = kmedoid.apply(dist, Vec(0, 1, 2, 3, 4, 5))

    show(kmedoid.plot(data, res.clusters, 2))

  }

}
