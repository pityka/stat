package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample}
import slogging.StrictLogging
import stat.sparse._
import stat._

case class Batch(x: Mat[Double], y: Vec[Double], penalizationMask: Vec[Double])
case class DataSource(
    training: Iterator[Iterator[Batch]],
    numCols: Int,
    allidx: Vec[Int],
    batchPerEpoch: Int
)

case class MatrixData(trainingX: Mat[Double],
                      trainingY: Vec[Double],
                      batchSize: Int,
                      penalizationMask: Vec[Double])

case class FrameData[RX: ST: ORD](f: Frame[RX, String, Double],
                                  yKey: String,
                                  missingMode: MissingMode = DropSample,
                                  addIntercept: Boolean = true,
                                  standardize: Boolean = false,
                                  batchSize: Int)

trait DataSourceFactory[T] {
  def apply(t: T,
            allowedIdx: Option[Vec[Int]],
            rng: scala.util.Random): DataSource
}

trait DataSourceFactories extends StrictLogging {

  implicit def matrixDataSource = new DataSourceFactory[MatrixData] {
    def apply(t: MatrixData,
              allowedIdx2: Option[Vec[Int]],
              rng: scala.util.Random): DataSource = {
      import t._

      val allowedIdx = allowedIdx2.getOrElse(0 until trainingX.numRows toVec)

      val fullBatch =
        if (batchSize >= allowedIdx.length)
          Some(
            Batch(trainingX.takeRows(allowedIdx),
                  trainingY(allowedIdx),
                  penalizationMask))
        else None

      val iter = new Iterator[Iterator[Batch]] {

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

                val idx2 = idx(
                  c to math.min(idx.length - 1, c + batchSize): _*)
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
      DataSource(iter,
                 trainingX.numCols,
                 (0 until trainingX.numRows).toVec,
                 allowedIdx.length / batchSize + 1)
    }

  }

  implicit def frameDataSource[RX: ST: ORD] =
    new DataSourceFactory[FrameData[RX]] {
      def apply(
          data: FrameData[RX],
          allowedIdx2: Option[Vec[Int]],
          rng: scala.util.Random
      ): DataSource = {
        import data._

        val (x, y, std) =
          createDesignMatrix(f, yKey, missingMode, addIntercept)

        implicitly[DataSourceFactory[MatrixData]].apply(
          MatrixData(trainingX = x,
                     trainingY = y,
                     batchSize = math.min(batchSize, x.numRows),
                     penalizationMask =
                       if (standardize)
                         std.toSeq.zipWithIndex
                           .map(x => if (x._2 == 0) 0d else 1.0 / x._1)
                           .toVec
                       else Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*)),
          allowedIdx = None,
          rng = rng
        )

      }

    }

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

  // def fromSparseMat(trainingX: SMat,
  //                   trainingY: SVec,
  //                   allowedIdx: Vec[Int],
  //                   batchSize: Int,
  //                   penalizationMask: Vec[Double],
  //                   seed: Int): DataSource = {
  //   val cidx = sparse.colIx(trainingX)
  //
  //   val iter = new Iterator[Iterator[Batch]] {
  //
  //     val rng = new scala.util.Random(seed)
  //
  //     def hasNext = true
  //     def next = {
  //       new Iterator[Batch] {
  //
  //         var c = 0
  //         val idx = rng.shuffle(allowedIdx.toSeq).toVec
  //
  //         def hasNext = c < idx.length - 1
  //         def next = {
  //
  //           val idx2 = idx(c to math.min(idx.length - 1, c + batchSize): _*)
  //           c += batchSize
  //
  //           logger.trace("New batch of size {} ", idx2.length)
  //
  //           Batch(sparse.dense(trainingX, idx2, cidx),
  //                 sparse.dense(trainingY, idx2),
  //                 penalizationMask)
  //         }
  //
  //       }
  //     }
  //   }
  //   DataSource(iter, cidx.length, allowedIdx.length / batchSize + 1)
  // }
}
