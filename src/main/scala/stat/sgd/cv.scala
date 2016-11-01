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
  EvalR
}

object Cv {

  def fitWithCV[I <: ItState, E](
      x: Mat[Double],
      y: Vec[Double],
      obj: ObjectiveFunction[E],
      pen: Penalty,
      upd: Updater[I],
      trainRatio: Double,
      split: CVSplit,
      hMin: Double,
      hMax: Double,
      hN: Int,
      penalizationMask: Vec[Double],
      maxIterations: Int,
      minEpochs: Int,
      convergedAverage: Int,
      epsilon: Double,
      seed: Int
  ): (EvalR[E], Vec[Double]) = {
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

    val nested = Train.nestGridSearch(training, split, hMin, hMax, hN)

    trainOnTestEvalOnHoldout(
      (0 until x.numRows).toVec,
      nested,
      Split(trainRatio, seed),
      Double.NaN
    ).next

  }

  def train[I <: ItState, E](
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
  ): Train[E] = new Train[E] {
    def train(idx: Vec[Int], hyper: Double): Eval[E] = {

      val result: SgdResult[E] = Sgd.optimize(
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
