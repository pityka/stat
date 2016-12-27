package stat.vis

import org.saddle._
import org.saddle.linalg._
import org.nspl._
import org.nspl.saddle._

object CorrelationPlot {

  def fromColumns[RX <: AnyRef: ST: ORD, CX <: AnyRef: ST: ORD](
      f: Frame[RX, CX, Double],
      main: String = "",
      xlab: String = "",
      ylab: String = "",
      xLabFontSize: Option[RelFontSize] = None,
      yLabFontSize: Option[RelFontSize] = None,
      mainFontSize: RelFontSize = 1 fts,
      colormap: Colormap = RedBlue(-1, 1, 0)) = {

    val vecs = f.toColSeq.map {
      case (cx, series) =>
        (cx, series.toVec.demeaned, series.toVec.stdev)
    }

    val ar = Array.ofDim[Double](vecs.size * vecs.size)

    var i = 0
    var j = 0
    while (i < vecs.size) {
      while (j < i) {
        val (c1, v1, s1) = vecs(i)
        val (c2, v2, s2) = vecs(j)
        val cov = v1 vv v2 * (1d / (v1.length - 1))
        val r = cov / (s1 * s2)
        ar(i * vecs.size + j) = r
        ar(j * vecs.size + i) = r
        j += 1
      }
      j = 0
      i += 1
    }

    val f2 = Frame(Mat(vecs.size, vecs.size, ar),
                   Index(vecs.map(_._1): _*),
                   Index(vecs.map(_._1): _*))

    val plot = Heatmap
      .fromColumns(
        frame = f2,
        reorderRows = true,
        euclidean = false,
        main = main,
        xlab = xlab,
        ylab = ylab,
        xLabFontSize = xLabFontSize,
        yLabFontSize = yLabFontSize,
        mainFontSize = mainFontSize,
        colormap = colormap
        // zlim = Some(-1d -> 1d)
      )
      ._1

    (plot, f2)

  }

}
