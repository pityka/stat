package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample, createDesignMatrix}
import slogging.StrictLogging

case class Batch(x: Mat[Double], y: Vec[Double], penalizationMask: Vec[Double])
case class DataSource(
    training: Iterator[Iterator[Batch]],
    numCols: Int,
    batchPerEpoch: Int
)

object DataSource extends StrictLogging {

  def createDesignMatrix[RX: ST: ORD](f: Frame[RX, String, Double],
                                      yKey: String,
                                      missingMode: MissingMode = DropSample,
                                      addIntercept: Boolean = true)
    : (Mat[Double], Vec[Double], Vec[Double]) = {
    val data2 =
      stat.regression.createDesignMatrix(f, missingMode, addIntercept)

    val x = data2.filterIx(_ != yKey)
    (x.toMat, data2.firstCol(yKey).toVec, x.stdev.toVec)
  }

  def fromFrame[RX: ST: ORD](f: Frame[RX, String, Double],
                             yKey: String,
                             missingMode: MissingMode = DropSample,
                             addIntercept: Boolean = true,
                             batchSize: Int = 256,
                             seed: Int = 42,
                             standardize: Boolean = true): DataSource = {

    val (x, y, std) =
      createDesignMatrix(f, yKey, missingMode, addIntercept)

    fromMat(
      trainingX = x,
      trainingY = y,
      allowedIdx = (0 until x.numRows).toVec,
      batchSize = math.min(batchSize, x.numRows),
      penalizationMask =
        if (standardize)
          std.toSeq.zipWithIndex
            .map(x => if (x._2 == 0) 0d else 1.0 / x._1)
            .toVec
        else Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*),
      seed = seed
    )

  }

  def fromMat(trainingX: Mat[Double],
              trainingY: Vec[Double],
              allowedIdx: Vec[Int],
              batchSize: Int,
              penalizationMask: Vec[Double],
              seed: Int): DataSource = {

    val fullBatch =
      if (batchSize >= allowedIdx.length)
        Some(
          Batch(trainingX.takeRows(allowedIdx),
                trainingY(allowedIdx),
                penalizationMask))
      else None

    val iter = new Iterator[Iterator[Batch]] {

      val rng = new scala.util.Random(seed)

      def hasNext = true
      def next = {
        new Iterator[Batch] {

          var c = 0
          val idx = rng.shuffle(allowedIdx.toSeq).toVec

          def hasNext = c < idx.length - 1
          def next = {
            if (fullBatch.isDefined) {
              c += batchSize
              logger.trace("Full batch of size {} (no copy)", idx.length)
              fullBatch.get
            } else {

              val idx2 = idx(c to math.min(idx.length - 1, c + batchSize): _*)
              c += batchSize

              logger.trace("New batch of size {} ", idx2.length)

              Batch(trainingX.takeRows(idx2),
                    trainingY(idx2),
                    penalizationMask)
            }

          }
        }
      }
    }
    DataSource(iter, trainingX.numCols, allowedIdx.length / batchSize + 1)
  }
}
