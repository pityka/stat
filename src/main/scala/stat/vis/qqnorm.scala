package stat.vis

import org.nspl._
import org.nspl.data._
import org.nspl.awtrenderer._
import org.nspl.saddle._
import org.saddle._
import org.saddle.linalg._

object QQNorm {

  def apply(data: Vec[Double]) = {
    val mean = data.mean2
    val std = data.sampleStandardDeviation
    val sorted = data.sorted
    val n = sorted.length.toDouble
    val qs = 1 until sorted.length map { i =>
      jdistlib.Normal.quantile(i / n, mean, std, true, false)
    } toVec

    xyplot(qs -> sorted.take(1 until sorted.length toArray))(
      xlab = "quantiles",
      ylab = "data",
      draw1Line = true,
      main = "Normal qq plot")
  }

}
