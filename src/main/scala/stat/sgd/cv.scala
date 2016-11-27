package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{MissingMode, DropSample, Prediction, NamedPrediction}
import stat.crossvalidation.{
  Train,
  Eval,
  CVSplit,
  trainOnTestEvalOnHoldout,
  Split,
  EvalR,
  HyperParameterSearch
}
import slogging.StrictLogging
import stat.matops._
object Cv {

  def fitWithCV[RX: ST: ORD, I <: ItState, E, H, P](
      data: Frame[RX, String, Double],
      yKey: String,
      missingMode: MissingMode,
      addIntercept: Boolean,
      standardize: Boolean,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      trainRatio: Double,
      split: CVSplit,
      search: HyperParameterSearch[H],
      hMin: Double,
      hMax: Double,
      hN: Int,
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[FrameData[RX], Mat[Double]])
    : (EvalR[E], NamedSgdResult[E, P]) = {

    val (eval, result) =
      fitWithCV(FrameData(data, yKey, missingMode, addIntercept, standardize),
                obj,
                pen,
                upd,
                trainRatio,
                split,
                search,
                hMin,
                hMax,
                hN,
                maxIterations,
                minEpochs,
                convergedAverage,
                epsilon,
                data.numRows,
                data.numRows,
                rng)(dsf)

    val idx =
      obj
        .adaptParameterNames(
          if (addIntercept)
            ("intercept" +: data.colIx.toSeq.filter(_ != yKey))
          else data.colIx.toSeq.filter(_ != yKey))
        .toIndex

    (eval, NamedSgdResult(result, idx))
  }

  def fitWithCV[D, I <: ItState, E, H, P, M](
      data: D,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      trainRatio: Double,
      split: CVSplit,
      search: HyperParameterSearch[H],
      hMin: Double,
      hMax: Double,
      hN: Int,
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double,
      batchSize: Int,
      maxEvalSize: Int,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D, M]): (EvalR[E], SgdResult[E, P]) = {
    import dsf.ops

    val training = train(data,
                         obj,
                         pen,
                         upd,
                         maxIterations,
                         minEpochs,
                         convergedAverage,
                         epsilon,
                         batchSize,
                         maxEvalSize,
                         rng)

    val nested = Train.nestedSearch(training, split, hMin, hMax, hN, search)

    val allidx = dsf.apply(data, None, batchSize, rng).allidx

    val (eval, estimates) = trainOnTestEvalOnHoldout(
      allidx,
      nested,
      Split(trainRatio, rng)
    ).next

    val prediction = SgdResult(estimates, obj)
    (eval, prediction)

  }

  def train[D, I <: ItState, E, H, M](
      data: D,
      obj: ObjectiveFunction[E, _],
      pen: Penalty[H],
      upd: Updater[I],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double,
      batchSize: Int,
      evalBatchSize: Int,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D, M]): Train[E, H] =
    new Train[E, H] with StrictLogging {
      import dsf.ops
      def train(idx: Vec[Int], hyper: H): Option[Eval[E]] = {
        logger.trace("Train on {}", idx.length)
        Sgd
          .optimize(dsf.apply(data, Some(idx), batchSize, rng),
                    obj,
                    pen.withHyperParameter(hyper),
                    upd,
                    maxIterations,
                    minEpochs,
                    convergedAverage,
                    epsilon)
          .map { result =>
            new Eval[E] {
              def eval(idx: Vec[Int]): EvalR[E] = {

                val batch: Batch[M] =
                  dsf(data,
                      Some(idx),
                      batchSize = math.min(idx.length, evalBatchSize),
                      rng).training.next.next

                val obj = result.evaluateFit(batch)
                val e = result.evaluateFit2(batch)
                logger.debug("Eval on {} out of {}: obj - {}, misc - {}",
                             math.min(idx.length, evalBatchSize),
                             idx.length,
                             obj,
                             e)
                EvalR(obj, e)

              }
              def estimatesV = result.estimatesV
            }
          }
      }

    }

}
