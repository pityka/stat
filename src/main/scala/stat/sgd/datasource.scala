package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample}
import slogging.StrictLogging
import stat.sparse._
import stat._
import stat.matops._
case class Batch[M](x: M, y: Vec[Double], penalizationMask: Vec[Double])
case class DataSource[M](
    training: Iterator[Iterator[Batch[M]]],
    numCols: Int,
    allidx: Vec[Int],
    batchPerEpoch: Int
)

case class MatrixData(trainingX: Mat[Double],
                      trainingY: Vec[Double],
                      penalizationMask: Vec[Double])

case class SparseMatrixData(trainingX: SMat,
                            trainingY: SVec,
                            penalizationMask: Vec[Double])

case class FrameData[RX: ST: ORD](f: Frame[RX, String, Double],
                                  yKey: String,
                                  missingMode: MissingMode = DropSample,
                                  addIntercept: Boolean = true,
                                  standardize: Boolean = false)

trait DataSourceFactory[T, M] {
  implicit def ops: MatOps[M]
  def apply(t: T,
            allowedIdx: Option[Vec[Int]],
            batchSize: Int,
            rng: scala.util.Random): DataSource[M]
}

trait DataSourceFactories extends StrictLogging {

  implicit def matrixDataSource =
    new DataSourceFactory[MatrixData, Mat[Double]] {
      implicit val ops = DenseMatOps
      def apply(t: MatrixData,
                allowedIdx2: Option[Vec[Int]],
                batchSize: Int,
                rng: scala.util.Random): DataSource[Mat[Double]] = {
        import t._

        val allowedIdx = allowedIdx2.getOrElse(0 until trainingX.numRows toVec)

        val fullBatch =
          if (batchSize >= allowedIdx.length)
            Some(
              Batch(trainingX.takeRows(allowedIdx),
                    trainingY(allowedIdx),
                    penalizationMask))
          else None

        val iter = new Iterator[Iterator[Batch[Mat[Double]]]] {

          def hasNext = true
          def next = {
            new Iterator[Batch[Mat[Double]]] {

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
    new DataSourceFactory[FrameData[RX], Mat[Double]] {
      implicit val ops = DenseMatOps

      def apply(
          data: FrameData[RX],
          allowedIdx2: Option[Vec[Int]],
          batchSize1: Int,
          rng: scala.util.Random
      ): DataSource[Mat[Double]] = {
        import data._

        val (x, y, std) =
          createDesignMatrix(f, yKey, missingMode, addIntercept)

        val batchSize = math.min(batchSize1, x.numRows)

        implicitly[DataSourceFactory[MatrixData, Mat[Double]]].apply(
          MatrixData(trainingX = x,
                     trainingY = y,
                     penalizationMask =
                       if (standardize)
                         std.toSeq.zipWithIndex
                           .map(x => if (x._2 == 0) 0d else 1.0 / x._1)
                           .toVec
                       else Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*)),
          allowedIdx = None,
          batchSize = batchSize,
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

  implicit def sparseMatrixDataSource =
    new DataSourceFactory[SparseMatrixData, SMat] {
      implicit val ops: MatOps[SMat] = ???

      def apply(t: SparseMatrixData,
                allowedIdx2: Option[Vec[Int]],
                batchSize: Int,
                rng: scala.util.Random): DataSource[SMat] = {
        import t._
        val cidx = sparse.colIx(trainingX)

        val allowedIdx =
          allowedIdx2.getOrElse(0 until sparse.numRows(trainingX) toVec)

        val iter = new Iterator[Iterator[Batch[SMat]]] {

          def hasNext = true
          def next = {
            new Iterator[Batch[SMat]] {

              var c = 0
              val idx = rng.shuffle(allowedIdx.toSeq).toVec

              def hasNext = c < idx.length - 1
              def next: Batch[SMat] = {

                val idx2 = idx(
                  c until math.min(idx.length - 1, c + batchSize): _*)
                c += batchSize

                logger.trace("New batch of size {} ", idx2.length)
                ???
                // Batch(sparse.dense(trainingX, idx2, cidx),
                //       sparse.dense(trainingY, idx2),
                //       penalizationMask)
              }

            }
          }
        }
        DataSource(iter,
                   cidx.length,
                   sparse.rowIx(trainingX),
                   allowedIdx.length / batchSize + 1)
      }
    }
}
