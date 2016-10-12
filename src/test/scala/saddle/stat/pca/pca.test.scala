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
    ).demeaned
    println(data)
    println(fromData(data))
    val covM = data.toMat mmt data.toMat
    val cov = Frame(covM, data.rowIx, data.rowIx)
    println(cov)
    println(fromCovariance(cov))
  }

}
