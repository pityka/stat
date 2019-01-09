package stat.rbf

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import org.nspl.awtrenderer._
import org.nspl._
import org.nspl.saddle._
import stat.regression._

class RBFSuite extends FunSuite {
  test("short") {
    val y = 0 until 1000 map (i => math.pow(i / 500d, 2d))
    val x = 0 until 1000 map (i => i / 500d)
    val x2 = 0 until 1000 map (i => i / 250d)
    val data1 = Frame("x" -> Series(x: _*), "y" -> Series(y: _*))
    val (centers, bandwidths) =
      RadialBasisFunctions
        .centers(data1.filterIx(_ != "y"), 10, 5, 10, scala.util.Random)

    val rbfFeatures = RadialBasisFunctions
      .makeFrame(data1, centers, bandwidths)
      .mapColIndex(_.toString)

    println(centers)
    println(bandwidths)
    println(rbfFeatures)

    val data2 = data1.rconcat(rbfFeatures).filterIx(_ != "x")
    val lm1 = LinearRegression.linearRegression(
      data = data2,
      yKey = "y",
      lambda = 0.1,
      addIntercept = true
    )
    println(lm1.table)

    val data3 = Frame("x" -> Series(x2: _*))

    val predicted =
      lm1.predictFrame(RadialBasisFunctions
                         .makeFrame(data3, centers, bandwidths)
                         .mapColIndex(_.toString),
                       intercept = true)

    val joinedWithPredicted =
      data3.rconcat(Frame("pred" -> predicted))

    show(
      xyplot(
        (joinedWithPredicted.col("x", "pred"),
         List(point(color = Color.black)),
         InLegend("pred")),
        (data1.col("x", "y").sortedRowsBy(_.first("x")),
         List(line(color = Color.red)),
         InLegend("orig"))
      )(xAxisMargin = 0.1)
    )
  }
}
