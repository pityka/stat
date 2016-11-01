package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{
  MissingMode,
  DropSample,
  createDesignMatrix,
  Prediction,
  NamedPrediction
}
import slogging.StrictLogging

trait EvaluateFit[E] extends Prediction {
  def evaluateFit(batch: Batch): Double
  def evaluateFit2(batch: Batch): E
}

trait ItState {
  def point: Vec[Double]
  def convergence: Double
}

trait Updater[I <: ItState] {
  def next(x: Vec[Double],
           b: Batch,
           obj: ObjectiveFunction[_],
           pen: Penalty,
           last: Option[I]): I
}

case class SgdResult[E](
    estimatesV: Vec[Double],
    model: ObjectiveFunction[E]
) extends Prediction
    with EvaluateFit[E] {
  def predict(v: Vec[Double]) = model.predict(estimatesV, v)
  def predict(m: Mat[Double]) = model.predict(estimatesV, m)
  def evaluateFit(b: Batch): Double = model.apply(estimatesV, b)
  def evaluateFit2(b: Batch): E = model.eval(estimatesV, b)
}

case class NamedSgdResult[E](
    raw: SgdResult[E],
    names: Index[String]
) extends NamedPrediction
    with Prediction
    with EvaluateFit[E] {
  def estimatesV = raw.estimatesV
  def predict(v: Vec[Double]) = raw.predict(v)
  def predict(m: Mat[Double]) = raw.predict(m)
  def evaluateFit(b: Batch): Double = raw.evaluateFit(b)
  def evaluateFit2(b: Batch): E = raw.evaluateFit2(b)
}

object Sgd extends StrictLogging {

  def optimize[RX: ST: ORD, I <: ItState, E](
      f: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction[E],
      pen: Penalty,
      upd: Updater[I],
      missingMode: MissingMode = DropSample,
      addIntercept: Boolean = true,
      maxIterations: Int = 100000,
      minEpochs: Int = 2,
      convergedAverage: Int = 50,
      epsilon: Double = 1E-6,
      seed: Int = 42,
      standardize: Boolean = true
  ): NamedSgdResult[E] = {
    val result = optimize(DataSource.fromFrame(f,
                                               yKey,
                                               missingMode,
                                               addIntercept,
                                               f.numRows,
                                               seed,
                                               standardize),
                          obj,
                          pen,
                          upd,
                          maxIterations,
                          minEpochs,
                          convergedAverage,
                          epsilon)

    NamedSgdResult(result, f.colIx.toSeq.filter(_ != yKey).toIndex)
  }

  def optimize[I <: ItState, E](
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction[E],
      pen: Penalty,
      upd: Updater[I],
      penalizationMask: Vec[Double],
      maxIterations: Int,
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): SgdResult[E] =
    optimize(DataSource.fromMat(x,
                                y,
                                (0 until x.numRows).toVec,
                                x.numRows,
                                penalizationMask,
                                seed),
             obj,
             pen,
             upd,
             maxIterations,
             minEpochs,
             convergedAverage,
             epsilon)

  def optimize[I <: ItState, E](
      dataSource: DataSource,
      obj: ObjectiveFunction[E],
      pen: Penalty,
      updater: Updater[I],
      maxIterations: Int,
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double
  ): SgdResult[E] = {

    val data: Iterator[Batch] = dataSource.training.flatten
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
        start #:: loop({
          val n = updater.next(start.point, data.next, obj, pen, Some(start))

          logger.trace(n.toString)

          n
        })

      val f1 = updater.next(start, data.next, obj, pen, None)
      loop(f1)
    }

    SgdResult(iteration(start,
                        maxIterations,
                        minEpochs * dataSource.batchPerEpoch,
                        convergedAverage,
                        epsilon),
              obj)

  }

}
