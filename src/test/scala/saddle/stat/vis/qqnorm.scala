package stat.vis

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import org.nspl._

class QQNormSute extends FunSuite {
  test("short") {
    val d = array.randNormal(1000)
    show(QQNorm(d))
  }

}
