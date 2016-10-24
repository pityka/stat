package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample, createDesignMatrix}

case class Batch(x: Mat[Double], y: Vec[Double], penalizationMask: Vec[Double])
case class DataSource(
    training: Iterator[Iterator[Batch]],
    numCols: Int
)

object DataSource {

  def fromFrame[RX: ST: ORD](f: Frame[RX, String, Double],
                             yKey: String,
                             missingMode: MissingMode = DropSample,
                             addIntercept: Boolean = true,
                             batchSize: Int = 256,
                             seed: Int = 42,
                             standardize: Boolean = true): DataSource = {

    val data2 =
      createDesignMatrix(f, missingMode, addIntercept)

    val x = data2.filterIx(_ != yKey)
    fromMat(
      trainingX = x.toMat,
      trainingY = data2.firstCol(yKey).toVec,
      batchSize = math.min(batchSize, data2.numRows),
      penalizationMask =
        if (standardize)
          x.stdev.toVec.toSeq.zipWithIndex
            .map(x => if (x._2 == 0) 0d else 1.0 / x._1)
            .toVec
        else Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*),
      seed = seed
    )

  }

  def fromMat(trainingX: Mat[Double],
              trainingY: Vec[Double],
              batchSize: Int,
              penalizationMask: Vec[Double],
              seed: Int): DataSource = {

    val iter = new Iterator[Iterator[Batch]] {

      def hasNext = true
      def next = {
        new Iterator[Batch] {

          var c = 0
          val idx = new scala.util.Random(seed)
            .shuffle((0 until trainingX.numRows).toVector)
            .toVec

          def hasNext = c < trainingX.numRows - 1
          def next = {
            val idx2 = idx(
              c to math.min(trainingX.numRows - 1, c + batchSize): _*)
            c += batchSize

            Batch(trainingX.takeRows(idx2), trainingY(idx2), penalizationMask)

          }
        }
      }
    }
    DataSource(iter, trainingX.numCols)
  }
}

case class SgdEstimate(estimates: Vec[Double])

trait ItState {
  def point: Vec[Double]
  def convergence: Double
}

trait Updater[I <: ItState] {
  def next(x: Vec[Double],
           b: Batch,
           obj: ObjectiveFunction,
           pen: Penalty,
           last: Option[I]): I
}

object Sgd {

  def optimize[RX: ST: ORD, I <: ItState](
      f: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction,
      pen: Penalty,
      upd: Updater[I],
      missingMode: MissingMode = DropSample,
      addIntercept: Boolean = true,
      maxIterations: Int = 10000,
      minIterations: Int = 100,
      convergedAverage: Int = 50,
      epsilon: Double = 1E-6,
      seed: Int = 42,
      standardize: Boolean = true
  ): SgdEstimate =
    optimize(DataSource.fromFrame(f,
                                  yKey,
                                  missingMode,
                                  addIntercept,
                                  math.min(256, f.numRows),
                                  seed,
                                  standardize),
             obj,
             pen,
             upd,
             maxIterations,
             minIterations,
             convergedAverage,
             epsilon)

  def optimize[I <: ItState](
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction,
      pen: Penalty,
      upd: Updater[I],
      penalizationMask: Vec[Double],
      maxIterations: Int,
      minIterations: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): SgdEstimate =
    optimize(
      DataSource
        .fromMat(x, y, math.min(256, x.numRows), penalizationMask, seed),
      obj,
      pen,
      upd,
      maxIterations,
      minIterations,
      convergedAverage,
      epsilon)

  def optimize[I <: ItState](
      dataSource: DataSource,
      obj: ObjectiveFunction,
      pen: Penalty,
      updater: Updater[I],
      maxIterations: Int,
      minIterations: Int,
      convergedAverage: Int,
      epsilon: Double
  ): SgdEstimate = {

    val data = dataSource.training.flatten
    val start = vec.zeros(dataSource.numCols)

    def iteration(start: Vec[Double],
                  max: Int,
                  min: Int,
                  tail: Int,
                  epsilon: Double): Vec[Double] = {
      val t = from(start)
        .take(max)
        .drop(min)
        .filter(_.convergence < epsilon)
        .take(tail)

      t.map(_.point).reduce(_ + _) / t.size

    }

    def from(start: Vec[Double]): Stream[I] = {

      def loop(start: I): Stream[I] =
        start #:: loop(
          updater.next(start.point, data.next, obj, pen, Some(start)))

      val f1 = updater.next(start, data.next, obj, pen, None)
      loop(f1)
    }

    SgdEstimate(
      iteration(start,
                maxIterations,
                minIterations,
                convergedAverage,
                epsilon))

  }

}
