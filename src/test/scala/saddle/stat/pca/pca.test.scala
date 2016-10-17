package stat.pca

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._

class PCASuite extends FunSuite {
  test("short") {
    val a: Vec[Double] = array.randDouble(50)
    val b: Vec[Double] = a + 2d
    val c: Vec[Double] = (array.randDouble(50): Vec[Double]) * 100d
    val data = Frame(
      Mat(a, b, c),
      Index((0 until 50).map(_.toString): _*),
      Index("a", "b", "c")
    )

    val covM = data.demeaned.toMat mmt data.demeaned.toMat
    val cov = Frame(covM, data.rowIx, data.rowIx)
    assert(
      fromData(data, 1).eigenvalues
        .roundTo(4) == fromCovariance(cov, 1).eigenvalues.roundTo(4))
  }

}
