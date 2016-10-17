package stat.vis

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import org.nspl._

class HeatmapSuite extends FunSuite {
  test("short") {
    val a: Vec[Double] = array.randDouble(50)
    val b: Vec[Double] = a + 2d
    val c: Vec[Double] = (array.randDouble(50): Vec[Double]) * 100d
    val data = Frame(
      Mat(a, b, c),
      Index((0 until 50).map(_.toString): _*),
      Index("a", "b", "c")
    ).demeaned
    val (heatmap, pairwise, distmat) =
      Heatmap.fromColumns(data.T, colormap = LogHeatMapColors())
    show(group(heatmap, pairwise, TableLayout(2)))

  }

}
