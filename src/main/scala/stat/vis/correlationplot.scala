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
      colormap: Colormap = RedBlue(-1, 1, 0),
      valueText: Boolean = false,
      valueColor: Color = Color.black,
      valueFontSize: RelFontSize = 0.4 fts,
      tickLength: RelFontSize = 0.0 fts) = {

    val vecs = f.toColSeq.map {
      case (cx, series) =>
        (cx, series.toVec)
    }

    def findNA(v: Array[Double]) = {
      val ab = scala.collection.mutable.ArrayBuffer[Int]()
      var i = 0
      val n = v.size
      while (i < n) {
        if (v(i).isNaN) {
          ab.append(i)
        }
        i += 1
      }
      ab.toArray
    }

    val ar = Array.ofDim[Double](vecs.size * vecs.size)

    var i = 0
    var j = 0
    while (i < vecs.size) {
      while (j < i) {
        val (c1, v1) = vecs(i)
        val (c2, v2) = vecs(j)
        val nanidx = findNA(v1) ++ findNA(v2)
        val v1wonan = v1.without(nanidx)
        val v2wonan = v2.without(nanidx)
        val v1dem = v1wonan.demeaned
        val v2dem = v2wonan.demeaned
        val v1s = v1dem.stdev
        val v2s = v2dem.stdev
        val cov = v1dem vv v2dem * (1d / (v1dem.length - 1))
        val r = cov / (v1s * v2s)
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
        colormap = colormap,
        valueText = valueText,
        valueColor = valueColor,
        valueFontSize = valueFontSize,
        tickLength = tickLength
        // zlim = Some(-1d -> 1d)
      )
      ._1

    (plot, f2)

  }

}
