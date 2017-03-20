package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{Prediction, NamedPrediction}
import stat.crossvalidation.{
  Train,
  Eval,
  CVSplit,
  trainOnTestEvalOnHoldout,
  trainOnTestEvalOnHoldout2,
  Split,
  EvalR,
  HyperParameterSearch,
  HyperParameter
}
import slogging.StrictLogging
import stat.matops._
object Cv {

  def fitWithCV[RX: ST: ORD, I <: ItState, E, H, P, K](
      data: Frame[RX, String, Double],
      yKey: String,
      addIntercept: Boolean,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      outerSplit: CVSplit,
      split: CVSplit,
      search: HyperParameterSearch[H, K],
      kernelFactory: FeatureMapFactory[K],
      bootstrapAggregate: Option[Int],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      batchSize: Option[Int],
      stop: StoppingCriterion,
      rng: scala.util.Random,
      normalize: Boolean,
      warmStart: Boolean
  ): (EvalR[E], NamedSgdResult[E, P]) = {

    val (x, y, std) =
      createDesignMatrix(data, yKey, addIntercept)

    val matrixData = MatrixData(trainingX = x,
                                trainingY = y,
                                penalizationMask =
                                  Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*))

    val (eval, result) =
      fitWithCV(matrixData,
                obj,
                pen,
                upd,
                outerSplit,
                split,
                search,
                kernelFactory,
                bootstrapAggregate,
                maxIterations,
                minEpochs,
                convergedAverage,
                stop,
                batchSize.getOrElse(data.numRows),
                data.numRows,
                rng,
                normalize,
                warmStart)

    val idx =
      obj
        .adaptParameterNames(
          if (addIntercept)
            ("intercept" +: data.colIx.toSeq.filter(_ != yKey))
          else data.colIx.toSeq.filter(_ != yKey))
        .toIndex

    (eval, NamedSgdResult(result, idx))
  }

  def fitWithCV[D, I <: ItState, E, H, P, M, K](
      data: D,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      outerSplit: CVSplit,
      split: CVSplit,
      search: HyperParameterSearch[H, K],
      kernelFactory: FeatureMapFactory[K],
      bootstrapAggregate: Option[Int],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      stop: StoppingCriterion,
      batchSize: Int,
      maxEvalSize: Int,
      rng: scala.util.Random,
      normalize: Boolean,
      warmStart: Boolean
  )(implicit dsf: DataSourceFactory[D, M]): (EvalR[E], SgdResult[E, P]) = {
    import dsf.ops

    val (normed, normalizer) = dsf.normalize(data)

    val d1 = if (normalize) normed else data
    val n1 = if (normalize) normalizer else vec.ones(normalizer.length)

    val training = {
      val t =
        train(d1,
              obj,
              pen,
              upd,
              kernelFactory,
              maxIterations,
              minEpochs,
              convergedAverage,
              stop,
              batchSize,
              maxEvalSize,
              rng)

      if (bootstrapAggregate.isDefined)
        stat.crossvalidation.bootstrapAggregate(t,
                                                bootstrapAggregate.get,
                                                SgdResultAggregator.make[E, P],
                                                rng)
      else t
    }

    val nested = Train.nestedSearch(training, split, search, warmStart)

    val allidx = dsf.getAllIdx(normed)

    val (eval, estimates, hyperp) = trainOnTestEvalOnHoldout2(
      allidx,
      nested,
      outerSplit
    ).next

    val prediction =
      SgdResult(estimates, obj, kernelFactory(hyperp.kernel), n1)
    (eval, prediction)

  }

  def train[D, I <: ItState, E, H, M, P, KH](
      data: D,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      kernel: FeatureMapFactory[KH],
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      stop: StoppingCriterion,
      batchSize: Int,
      evalBatchSize: Int,
      rng: scala.util.Random
  )(implicit dsf: DataSourceFactory[D, M])
    : Train[E, H, SgdResultWithErrors[E, P], KH] =
    new Train[E, H, SgdResultWithErrors[E, P], KH] with StrictLogging {
      import dsf.ops
      def eval(result: SgdResultWithErrors[E, P]) = new Eval[E] {
        def eval(idx: Vec[Int]): EvalR[E] = {

          if (result.validationErrorPerSample.isDefined)
            EvalR(result.validationErrorPerSample.get._1,
                  result.validationErrorPerSample.get._2)
          else {
            logger.debug(
              "Validation error was not precomputed. Doing in Eval#eval. ")
            val batch: Batch[M] =
              dsf(data,
                  Some(idx),
                  batchSize = math.min(idx.length, evalBatchSize),
                  rng).training.next.next

            val obj: Double =
              result.result.unpenalizedObjectivePerSample(batch)
            val e: E = result.result.evaluateFit(batch)

            logger.debug(
              "Eval  on {} out of {}: obj - {}, misc - {}",
              math.min(idx.length, evalBatchSize),
              idx.length,
              obj,
              e
            )
            EvalR(obj, e)
          }

        }
        def estimatesV = result.result.estimatesV
      }
      def train(
          idx: Vec[Int],
          hyper: HyperParameter[H, KH],
          evalIdx: Option[Vec[Int]],
          start: Option[Vec[Double]]): Option[SgdResultWithErrors[E, P]] = {
        logger.trace("Train on {}", idx.length)
        val dataSource = dsf.apply(data, Some(idx), batchSize, rng)
        val validationBatch =
          evalIdx.map { evalIdx =>
            dsf
              .apply(data, Some(evalIdx), evalIdx.length, rng)
              .training
              .next
              .next
          }

        Sgd.optimize(dataSource,
                     obj,
                     pen.withHyperParameter(hyper.penalty),
                     upd,
                     kernel(hyper.kernel),
                     maxIterations,
                     minEpochs,
                     convergedAverage,
                     stop,
                     validationBatch,
                     start)

      }

    }

}
