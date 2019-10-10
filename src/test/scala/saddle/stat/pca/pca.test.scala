package stat.pca

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._

class PCASuite extends FunSuite {
  test("short") {
    val a: Vec[Double] = array.randDouble(50).toVec
    val b: Vec[Double] = a + 2d
    val c: Vec[Double] = (array.randDouble(50).toVec: Vec[Double]) * 100d
    val data = Frame(
      Mat(a, b, c),
      Index((0 until 50).map(_.toString): _*),
      Index("a", "b", "c")
    )

    val covM = data.mapVec(_.demeaned).toMat.outerM
    val cov = Frame(covM, data.rowIx, data.rowIx)
    assert(
      fromData(data, 1).eigenvalues
        .roundTo(4) == fromCovariance(cov, 1).eigenvalues.roundTo(4))

    show(
      plot(fromData(data, 2),
           2,
           scala.util.Left(
             data
               .colAt(0)
               .mapValues(x => (x > 0).toString)
               .toSeq
               .drop(3)
               .toSeries)))
  }

}
