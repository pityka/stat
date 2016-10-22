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

    // val (validationIdx, trainingIdx) = scala.util.Random
    //   .shuffle((0 until x.numRows).toVector)
    //   .splitAt(validationSize)
    //
    // val validation =
    //   Batch(x.takeRows(validationIdx: _*), y.apply(validationIdx: _*))
    //
    // val (trainingX, trainingY) =
    //   (x.takeRows(trainingIdx: _*), y.app(trainingIdx: _*))

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

trait ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
}

trait Penalty {

  def proximal(b: Vec[Double], batch: Batch): Vec[Double]

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double]
  def hessian(p: Vec[Double], batch: Batch): Mat[Double]
}

object LinearRegression extends ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = Mat(y) - (X mm Mat(b))
    (yMinusXb tmm X).row(0)

  }

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] = {
    batch.x.innerM * (-1)
  }
}

case class L2(lambda: Double) extends Penalty {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    b * batch.penalizationMask * lambda

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    mat.diag(batch.penalizationMask * lambda)

  def proximal(w: Vec[Double], batch: Batch): Vec[Double] =
    w.zipMap(batch.penalizationMask)((old, pw) => old / (1.0 + pw * lambda))
}

//     def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
// // proximal gradient descent, fix this redundancy
//
//
//       (hessian(b, batch) mm Mat(
//         b - prox(b - (hessian(b, batch).invert mm o.jacobi(b, batch)).col(0))))
//         .col(0)
//     }
case class L1(lambda: Double) extends Penalty {

  def proximal(w: Vec[Double], batch: Batch) =
    w.zipMap(batch.penalizationMask)((old, pw) =>
      math.signum(old) * math.max(0.0, math.abs(old) - lambda * pw))

  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] =
    b.zipMap(batch.penalizationMask)((b, p) =>
      (if (b == 0) 0d else if (b < 0) -1 * lambda * p else lambda * p))

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] =
    mat.zeros(p.length, p.length)

}

case class SgdEstimate(estimates: Vec[Double])

case class Iteration(point: Vec[Double], lprimesum: Double)

trait Updater {
  def next(x: Vec[Double],
           b: Batch,
           obj: ObjectiveFunction,
           pen: Penalty): Iteration
}

object NewtonUpdater extends Updater {
  def next(b: Vec[Double],
           batch: Batch,
           obj: ObjectiveFunction,
           pen: Penalty) = {

    val j = obj.jacobi(b, batch) - pen.jacobi(b, batch)
    val h = obj.hessian(b, batch) - pen.hessian(b, batch)

    val hinv =
      (h * (-1)).invertPD
        .map(_ * (-1))
        .getOrElse(h.invertPD.getOrElse(h.invert))

    val next = b - (hinv mm j).col(0)

    val jacobisum =
      (obj.jacobi(next, batch) - pen.jacobi(next, batch)).map(math.abs).sum

    val r = Iteration(next, jacobisum)
    println(r)
    r
  }
}

object Sgd {

  def optimize[RX: ST: ORD](
      f: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction,
      pen: Penalty,
      upd: Updater,
      missingMode: MissingMode = DropSample,
      addIntercept: Boolean = true,
      maxIterations: Int = 10000,
      minIterations: Int = 100,
      convergedAverage: Int = 50,
      epsilon: Double = 1E-8,
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

  def optimize(
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction,
      pen: Penalty,
      upd: Updater,
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

  def optimize(
      dataSource: DataSource,
      obj: ObjectiveFunction,
      pen: Penalty,
      updater: Updater,
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
        .filter(_.lprimesum < epsilon)
        .take(tail)

      t.map(_.point).reduce(_ + _) / t.size

    }

    def from(start: Vec[Double]): Stream[Iteration] = {

      def loop(start: Iteration): Stream[Iteration] =
        start #:: loop(updater.next(start.point, data.next, obj, pen))

      val f1 = updater.next(start, data.next, obj, pen)
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
