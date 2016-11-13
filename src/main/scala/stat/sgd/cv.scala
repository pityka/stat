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
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): (EvalR[E], NamedSgdResult[E, P]) = {
    val (x, y, std) =
      DataSource.createDesignMatrix(data, yKey, missingMode, addIntercept)

    val penalizationMask =
      if (standardize)
        std.toSeq.zipWithIndex
          .map(x => if (x._2 == 0) 0d else 1.0 / x._1)
          .toVec
      else Vec(0d +: vec.ones(x.numCols - 1).toSeq: _*)

    val (eval, result) = fitWithCV(x,
                                   y,
                                   obj,
                                   pen,
                                   upd,
                                   trainRatio,
                                   split,
                                   search,
                                   hMin,
                                   hMax,
                                   hN,
                                   penalizationMask,
                                   maxIterations,
                                   minEpochs,
                                   convergedAverage,
                                   epsilon,
                                   seed)

    val idx =
      obj
        .adaptParameterNames(
          if (addIntercept)
            ("intercept" +: data.colIx.toSeq.filter(_ != yKey))
          else data.colIx.toSeq.filter(_ != yKey))
        .toIndex

    (eval, NamedSgdResult(result, idx))
  }

  def fitWithCV[I <: ItState, E, H, P](
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction[E, P],
      pen: Penalty[H],
      upd: Updater[I],
      trainRatio: Double,
      split: CVSplit,
      search: HyperParameterSearch[H],
      hMin: Double,
      hMax: Double,
      hN: Int,
      penalizationMask: Vec[Double],
      maxIterations: Int,
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): (EvalR[E], SgdResult[E, P]) = {
    val training = train(x,
                         y,
                         obj,
                         pen,
                         upd,
                         penalizationMask,
                         maxIterations,
                         minEpochs,
                         convergedAverage,
                         epsilon,
                         seed)

    val nested = Train.nestedSearch(training, split, hMin, hMax, hN, search)

    val (eval, estimates) = trainOnTestEvalOnHoldout(
      (0 until x.numRows).toVec,
      nested,
      Split(trainRatio, seed)
    ).next

    val prediction = SgdResult(estimates, obj)
    (eval, prediction)

  }

  def train[I <: ItState, E, H](
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction[E, _],
      pen: Penalty[H],
      upd: Updater[I],
      penalizationMask: Vec[Double],
      maxIterations: Int,
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): Train[E, H] = new Train[E, H] {
    def train(idx: Vec[Int], hyper: H): Eval[E] = {

      val result = Sgd.optimize(
        DataSource.fromMat(x, y, idx, x.numRows, penalizationMask, seed),
        obj,
        pen.withHyperParameter(hyper),
        upd,
        maxIterations,
        minEpochs,
        convergedAverage,
        epsilon)
      new Eval[E] {
        def eval(idx: Vec[Int]): EvalR[E] = {
          val batch: Batch = DataSource
            .fromMat(x, y, idx, x.numRows, penalizationMask, seed)
            .training
            .next
            .next

          val obj = result.evaluateFit(batch)
          val e = result.evaluateFit2(batch)
          EvalR(obj, e)

        }
        def estimatesV = result.estimatesV
      }
    }

  }

}
