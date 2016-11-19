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
  )(implicit dsf: DataSourceFactory[FrameData[RX]])
    : (EvalR[E], NamedSgdResult[E, P]) = {

    val (eval, result) = fitWithCV(FrameData(data,
                                             yKey,
                                             missingMode,
                                             addIntercept,
                                             standardize,
                                             data.numRows),
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
                                   rng)

    val idx =
      obj
        .adaptParameterNames(
          if (addIntercept)
            ("intercept" +: data.colIx.toSeq.filter(_ != yKey))
          else data.colIx.toSeq.filter(_ != yKey))
        .toIndex

    (eval, NamedSgdResult(result, idx))
  }

  def fitWithCV[D, I <: ItState, E, H, P](
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
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D]): (EvalR[E], SgdResult[E, P]) = {
    val training = train(data,
                         obj,
                         pen,
                         upd,
                         maxIterations,
                         minEpochs,
                         convergedAverage,
                         epsilon,
                         rng)

    val nested = Train.nestedSearch(training, split, hMin, hMax, hN, search)

    val allidx = dsf.apply(data, None, rng).allidx

    val (eval, estimates) = trainOnTestEvalOnHoldout(
      allidx,
      nested,
      Split(trainRatio, rng)
    ).next

    val prediction = SgdResult(estimates, obj)
    (eval, prediction)

  }

  def train[D, I <: ItState, E, H](
      data: D,
      obj: ObjectiveFunction[E, _],
      pen: Penalty[H],
      upd: Updater[I],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      epsilon: Double,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D]): Train[E, H] = new Train[E, H] {
    def train(idx: Vec[Int], hyper: H): Option[Eval[E]] = {

      Sgd
        .optimize(dsf.apply(data, Some(idx), rng),
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
              val batch: Batch = dsf(data, Some(idx), rng).training.next.next

              val obj = result.evaluateFit(batch)
              val e = result.evaluateFit2(batch)
              EvalR(obj, e)

            }
            def estimatesV = result.estimatesV
          }
        }
    }

  }

}
