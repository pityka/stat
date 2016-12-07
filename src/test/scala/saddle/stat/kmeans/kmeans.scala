package stat.kmeans

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import stat.sparse._

class KMeansSuite extends FunSuite {
  test("short") {
    val data = mat.randn(10000, 2)
    val res = kmeans.apply(data, mat.randn(18, 2), 10)

    show(plot(matToSparse(data), res, 2))

  }

  test("sparse") {
    val ncol = 3
    val nclust = 6
    val data = 0 until 1000 map { i =>
      val v = 0 until ncol map { j =>
        j -> scala.util.Random.nextDouble
      } filter (_._2 >= 0.8) map (i => i._1 -> scala.util.Random.nextGaussian)
      SVec(v.toSeries, ncol)
    }
    val res = kmeans.apply(data, mat.randn(nclust, ncol).rows.toVector, 50)
    show(plot(data, res, 3))
  }

  test("sparse big") {
    val ncol = 100000
    val nclust = 1000
    val data = 0 until 1000 map { i =>
      val v = 0 until ncol map { j =>
        j -> scala.util.Random.nextDouble
      } filter (_._2 >= 0.9999) map (i =>
                                       i._1 -> scala.util.Random.nextGaussian)
      SVec(v.toSeries, ncol)
    }
    val res = kmeans.apply(data, mat.randn(nclust, ncol).rows.toVector, 50)
    // show(plot(data, res, 3))
  }

}
