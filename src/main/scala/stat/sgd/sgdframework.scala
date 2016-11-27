package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample, Prediction, NamedPrediction}
import slogging.StrictLogging
import stat.matops._

trait EvaluateFit[E, @specialized(Double) P] extends Prediction[P] {
  def evaluateFit[T: MatOps](batch: Batch[T]): Double
  def evaluateFit2[T: MatOps](batch: Batch[T]): E
}

trait ItState {
  def point: Vec[Double]
  def convergence: Double
}

trait Updater[I <: ItState] {
  def next[T: MatOps](x: Vec[Double],
                      b: Batch[T],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[I]): I
}

case class SgdResult[E, P](
    estimatesV: Vec[Double],
    model: ObjectiveFunction[E, P]
) extends Prediction[P]
    with EvaluateFit[E, P] {
  def predict(v: Vec[Double]) = predict(Mat(v).T).raw(0)
  def predict(m: Mat[Double]) = model.predictMat(estimatesV, m)
  def predict[T: MatOps](m: T) = model.predict(estimatesV, m)
  def evaluateFit[T: MatOps](b: Batch[T]): Double = model.apply(estimatesV, b)
  def evaluateFit2[T: MatOps](b: Batch[T]): E = model.eval(estimatesV, b)
}

case class NamedSgdResult[E, P](
    raw: SgdResult[E, P],
    names: Index[String]
) extends NamedPrediction[P]
    with Prediction[P]
    with EvaluateFit[E, P] {
  def estimatesV = raw.estimatesV
  def predict(v: Vec[Double]) = raw.predict(v)
  def predict(m: Mat[Double]) = raw.predict(m)
  def predict[T: MatOps](m: T) = raw.predict(m)

  def evaluateFit[T: MatOps](b: Batch[T]): Double = raw.evaluateFit(b)
  def evaluateFit2[T: MatOps](b: Batch[T]): E = raw.evaluateFit2(b)
}

object Sgd extends StrictLogging {

  def optimize[RX: ST: ORD, I <: ItState, E, P](
      f: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_],
      upd: Updater[I],
      missingMode: MissingMode = DropSample,
      addIntercept: Boolean = true,
      maxIterations: Int = 100000,
      minEpochs: Double = 2d,
      convergedAverage: Int = 50,
      epsilon: Double = 1E-6,
      standardize: Boolean = true,
      rng: scala.util.Random = new scala.util.Random(42)
  )(implicit sdf: DataSourceFactory[FrameData[RX], Mat[Double]])
    : Option[NamedSgdResult[E, P]] =
    optimize(FrameData(f, yKey, missingMode, addIntercept, standardize),
             obj,
             pen,
             upd,
             maxIterations,
             minEpochs,
             convergedAverage,
             epsilon,
             f.numRows,
             rng)(DenseMatOps, sdf).map { result =>
      val idx =
        obj
          .adaptParameterNames(
            if (addIntercept)
              ("intercept" +: f.colIx.toSeq.filter(_ != yKey))
            else f.colIx.toSeq.filter(_ != yKey))
          .toIndex
      NamedSgdResult(result, idx)
    }

  def optimize[D, I <: ItState, E, P, M: MatOps](
      data: D,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_],
      upd: Updater[I],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double,
      batchSize: Int,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D, M]): Option[SgdResult[E, P]] =
    optimize(dsf.apply(data, None, batchSize, rng),
             obj,
             pen,
             upd,
             maxIterations,
             minEpochs,
             convergedAverage,
             epsilon)

  def optimize[I <: ItState, E, P, M: MatOps](
      dataSource: DataSource[M],
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_],
      updater: Updater[I],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double
  ): Option[SgdResult[E, P]] = {

    val data: Iterator[Batch[M]] = dataSource.training.flatten
    val start = obj.start(dataSource.numCols)

    def iteration(start: Vec[Double],
                  max: Int,
                  min: Int,
                  tail: Int,
                  epsilon: Double): Option[Vec[Double]] = {
      val t = from(start).zipWithIndex
        .take(max)
        .drop(min)
        .filter(_._1.convergence < epsilon)
        .take(tail)
        .toVector

      if (t.isEmpty) {
        logger.warn("Did not converge after {} iterations", max)
        None
      } else {
        logger.debug("Converged after {} iterations", t.last._2 + 1)

        Some(t.map(_._1.point).reduce(_ + _) / t.size)
      }

    }

    def from(start: Vec[Double]): Stream[I] = {

      def loop(start: I): Stream[I] =
        start #:: loop({
          val n = updater.next(start.point, data.next, obj, pen, Some(start))

          logger.trace(n.toString)

          n
        })

      val f1 = updater.next(start, data.next, obj, pen, None)
      loop(f1)
    }

    iteration(start,
              maxIterations,
              (minEpochs * dataSource.batchPerEpoch).toInt + 1,
              convergedAverage,
              epsilon).map { r =>
      SgdResult(r, obj)
    }

  }

}
