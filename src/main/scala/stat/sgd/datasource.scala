package stat.sgd

import org.saddle._
import org.saddle.linalg._
import slogging.StrictLogging
import stat.sparse._
import stat._
import stat.matops._
import stat.crossvalidation._

case class Batch[M](x: M,
                    y: Vec[Double],
                    penalizationMask: Vec[Double],
                    full: Boolean)

object Batch {
  def apply[M: MatOps](b: Batch[M], k: FeatureMap): Batch[M] =
    Batch(k.applyMat(b.x),
          b.y,
          k.applyPenalizationMask(b.penalizationMask),
          b.full)
}

case class DataSource[M](
    training: Iterator[Iterator[Batch[M]]],
    allidx: Vec[Int],
    batchPerEpoch: Int
)

case class MatrixData(trainingX: Mat[Double],
                      trainingY: Vec[Double],
                      penalizationMask: Vec[Double])

case class SparseMatrixData(trainingX: SMat,
                            trainingY: Vec[Double],
                            penalizationMask: Vec[Double])

trait DataSourceFactory[T, M] {
  implicit def ops: MatOps[M]
  def apply(t: T,
            allowedIdx: Option[Vec[Int]],
            batchSize: Int,
            rng: scala.util.Random): DataSource[M]

  def getAllIdx(t: T): Vec[Int]

  def normalize(t: T): (T, Vec[Double])
}

trait DataSourceFactories extends StrictLogging {

  implicit def matrixDataSource =
    new DataSourceFactory[MatrixData, Mat[Double]] {
      implicit val ops = DenseMatOps

      def stdev(t: MatrixData): Vec[Double] =
        t.trainingX.cols
          .map(_.sampleStandardDeviation)
          .map(x => if (x == 0.0) 1.0 else 1d / x)
          .toVec

      def normalize(t: MatrixData) = {
        val std = stdev(t)
        val scaledTrainingX =
          t.trainingX.mDiagFromRight(std)
        (MatrixData(scaledTrainingX, t.trainingY, t.penalizationMask), std)
      }

      def getAllIdx(t: MatrixData) = 0 until t.trainingX.numRows toVec

      def apply(t: MatrixData,
                allowedIdx2: Option[Vec[Int]],
                batchSize: Int,
                rng: scala.util.Random): DataSource[Mat[Double]] = {
        import t._

        // logger.trace("Creating new data source. Size {} , {}",
        //              allowedIdx2.map(_.length),
        //              allowedIdx2.map(_.toSeq))

        val allowedIdx = allowedIdx2.getOrElse(0 until trainingX.numRows toVec)

        val fullBatch =
          if (batchSize >= allowedIdx.length)
            Some(
              Batch(trainingX.takeRows(allowedIdx),
                    trainingY(allowedIdx),
                    penalizationMask,
                    true))
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
                        penalizationMask,
                        false)
                }

              }
            }
          }
        }
        DataSource(iter,
                   (0 until trainingX.numRows).toVec,
                   allowedIdx.length / batchSize + 1)
      }

    }

  def createDesignMatrix[RX: ST: ORD](f: Frame[RX, String, Double],
                                      yKey: String,
                                      addIntercept: Boolean = true)
    : (Mat[Double], Vec[Double], Vec[Double]) = {
    val data2 =
      stat.regression.createDesignMatrix(f, addIntercept)

    val x = data2.filterIx(_ != yKey)
    (x.toMat, data2.firstCol(yKey).toVec, x.reduce(_.toVec.sampleStandardDeviation).toVec)
  }

  implicit def sparseMatrixDataSource =
    new DataSourceFactory[SparseMatrixData, SMat] {
      implicit val ops: MatOps[SMat] = SparseMatOps

      /* todo provide a meaningful implementation here */
      def stdev(t: SparseMatrixData) = vec.ones(sparse.numCols(t.trainingX))

      def normalize(t: SparseMatrixData) = (t, stdev(t))

      def getAllIdx(t: SparseMatrixData) = 0 until t.trainingX.size toVec

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

                Batch(idx2.map(i => trainingX(i)).toSeq.toIndexedSeq,
                      trainingY(idx2),
                      penalizationMask,
                      batchSize >= allowedIdx.length)
              }

            }
          }
        }
        DataSource(iter,
                   sparse.rowIx(trainingX),
                   allowedIdx.length / batchSize + 1)
      }
    }
}
