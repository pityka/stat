package stat.sgd

import org.saddle._
import org.saddle.linalg._

case class Batch(x: Mat[Double], y: Vec[Double])
case class DataSource(
    training: Iterator[Iterator[Batch]],
    numCols: Int
)

object DataSource {
  def fromMat(trainingX: Mat[Double], trainingY: Vec[Double], batchSize: Int) = {

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
          val idx = scala.util.Random
            .shuffle((0 until trainingX.numRows).toVector)
            .toVec

          def hasNext = c < trainingX.numRows - 1
          def next = {
            val idx2 = idx(
              c to math.min(trainingX.numRows - 1, c + batchSize): _*)
            c += batchSize

            Batch(trainingX.takeRows(idx2), trainingY(idx2))

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

object LinearRegression extends ObjectiveFunction {
  def jacobi(b: Vec[Double], batch: Batch): Vec[Double] = {
    val y = batch.y
    val X = batch.x
    val yMinusXb = Mat(y) - (X mm Mat(b))
    (yMinusXb tmm X).row(0)

  }

  def hessian(p: Vec[Double], batch: Batch): Mat[Double] = {
    batch.x.innerM
  }
}

case class SgdEstimate(estimates: Vec[Double])

object Sgd {

  def optimize(
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction,
      maxIterations: Int = 10000,
      minIterations: Int = 100,
      convergedAverage: Int = 20,
      epsilon: Double = 1E-6
  ): SgdEstimate =
    optimize(DataSource.fromMat(x, y, math.min(256, x.numRows)),
             obj,
             maxIterations,
             minIterations,
             convergedAverage,
             epsilon)

  def optimize(
      dataSource: DataSource,
      obj: ObjectiveFunction,
      maxIterations: Int,
      minIterations: Int,
      convergedAverage: Int,
      epsilon: Double
  ): SgdEstimate = {

    case class Iteration(point: Vec[Double], lprimesum: Double)

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
        start #:: loop(nextGuess(start.point, data.next))

      val f1 = nextGuess(start, data.next)
      loop(f1)
    }

    def nextGuess(b: Vec[Double], batch: Batch) = {
      val j = obj.jacobi(b, batch)
      val h = obj.hessian(b, batch)
      val hinv = h.invertPD
      val next = b + (hinv mm j).col(0)
      val jacobisum = obj.jacobi(next, batch).map(math.abs).sum
      val r = Iteration(next, jacobisum)
      r
    }

    SgdEstimate(
      iteration(start,
                maxIterations,
                minIterations,
                convergedAverage,
                epsilon))

  }

}
