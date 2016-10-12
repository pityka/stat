package stat.pca

import org.saddle._
import org.scalatest.FunSuite
import stat._

class PCASuite extends FunSuite {
  test("fromData") {
    val a: Vec[Double] = array.randDouble(50)
    val b: Vec[Double] = a + 2d
    val c: Vec[Double] = array.randDouble(50)
    val data = Frame(
      Mat(a, b, c),
      Index((0 until 50).map(_.toString): _*),
      Index("a", "b", "c")
    )
    println(fromData(data))
  }
}
