package stat.rbf

import org.saddle._
import org.saddle.linalg._
import stat.matops._

object RadialBasisFunctions {

  def euclid(v1: Vec[Double], v2: Vec[Double]) = {
    val d = v1 - v2
    // println(v1.toString + " " + v2)
    // println(math.sqrt(d vv d))
    math.sqrt(d vv d)
  }

  def gaussian(variance: Double) =
    (r: Double) => math.exp(r * r * (-1) / (2 * variance))

  def makeRow(data: Vec[Double],
              centers: Seq[(Vec[Double], Double)]): Vec[Double] =
    centers.map(center => gaussian(center._2)(euclid(center._1, data))).toVec

  def makeFrame[RX: ST: ORD, CX: ST: ORD, RX2: ST: ORD](
      f1: Frame[RX, CX, Double],
      centers: Frame[RX2, CX, Double],
      bandwidths: Series[RX2, Double]): Frame[RX, RX2, Double] = {

    val reindexedF =
      f1.reindexCol(centers.colIx)

    val centerVecs: Seq[(RX2, (Series[CX, Double], Double))] =
      centers.toRowSeq.map {
        case (rx2, series) =>
          (rx2, (series, bandwidths.first(rx2).get))
      }
    val centerIndex: Index[RX2] = Index(centerVecs.map(_._1): _*)

    Frame(reindexedF.toRowSeq.map {
      case (rx, series) =>
        val v: Vec[Double] =
          makeRow(series.toVec, centerVecs.map(x => x._2._1.toVec -> x._2._2))
        (rx, Series(v, centerIndex))
    }: _*).T
  }

  def centers[RX: ST: ORD, CX: ST: ORD](
      data: Frame[RX, CX, Double],
      init: Seq[RX]): (Frame[RX, CX, Double], Series[RX, Double]) = {
    val dataM = data.toMat
    val centers = data.row(init: _*).toMat.rows
    val avgDist = Series(centers.zip(init).map {
      case (c, irx) =>
        irx -> math.pow(
          centers.map(c2 => euclid(c, c2)).filterNot(_ == 0d).min,
          2d) * 0.5
    }: _*)

    val centersF = Frame(centers.zip(init).map {
      case (c, irx) =>
        (irx, Series(c, data.colIx))
    }: _*).T

    (centersF, avgDist)
  }

  def centers[RX: ST: ORD, CX: ST: ORD](data: Frame[RX, CX, Double],
                                        clusters: Int,
                                        iter: Int,
                                        restarts: Int,
                                        rng: scala.util.Random)
    : (Frame[String, CX, Double], Series[String, Double]) = {

    val centers = stat.kmeans
      .random(stat.kmeans.matToSparse(data.toMat),
              clusters,
              restarts,
              iter,
              rng)
      .means
    val avgDist = Series(centers.map { c =>
      ("rbf_" + c.hashCode.toString) -> math
        .pow(centers.map(c2 => euclid(c, c2)).filterNot(_ == 0d).min, 2d) * 0.5
    }: _*)

    val centersF = Frame(centers.map { c =>
      (("rbf_" + c.hashCode.toString), Series(c, data.colIx))
    }: _*).T

    (centersF, avgDist)
  }

  def centers(data: Mat[Double],
              clusters: Int,
              iter: Int,
              restarts: Int,
              rng: scala.util.Random): (Seq[Vec[Double]], Vec[Double]) = {

    val centers = stat.kmeans
      .random(stat.kmeans.matToSparse(data), clusters, restarts, iter, rng)
      .means

    val avgDist = centers.map { c =>
      math.pow(centers.map(c2 => euclid(c, c2)).filterNot(_ == 0d).min, 2d)
    }.toVec

    (centers, avgDist)
  }

  def makeMat(f1: Mat[Double],
              centers: Seq[Vec[Double]],
              bandwidths: Vec[Double]): Mat[Double] = {

    val centerVecs =
      centers.zip(bandwidths.toSeq)

    Mat(f1.rows.map { row =>
      makeRow(row, centerVecs)
    }: _*).T

  }

}
