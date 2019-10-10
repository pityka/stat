package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.regression.{Prediction, NamedPrediction}
import slogging.StrictLogging
import stat.matops._
import stat.io.upicklers._
import upickle.default._
import ujson.Js

trait StoppingCriterion {
  def continue[I](s: Step[I]): Boolean
}

case class AbsoluteStopTraining(eps: Double) extends StoppingCriterion {
  def continue[I](s: Step[I]): Boolean =
    math.abs(s.penalizedObjectivePerSample) > eps
}

case class AbsoluteStopValidation(eps: Double) extends StoppingCriterion {
  def continue[I](s: Step[I]): Boolean =
    if (s.validationErrorPerSample.isEmpty)
      math.abs(s.penalizedObjectivePerSample) > eps
    else math.abs(s.validationErrorPerSample.get) > eps
}

case class RelativeStopValidation(eps: Double) extends StoppingCriterion {
  def continue[I](s: Step[I]): Boolean =
    if (s.relValE.isEmpty) math.abs(s.relObj) > eps
    else math.abs(s.relValE.get) > eps
}

case class RelativeStopTraining(eps: Double) extends StoppingCriterion {
  def continue[I](s: Step[I]): Boolean = math.abs(s.relObj) > eps
}

case class StopAfterFixedIterations(i: Int) extends StoppingCriterion {
  def continue[I](s: Step[I]): Boolean = s.count < i
}

case class TrainingStall(i: Int, eps: Double) extends StoppingCriterion {
  var max: Double = Double.NegativeInfinity
  var maxi: Int = 0
  var c = 0
  def continue[I](s: Step[I]): Boolean = {
    c += 1
    val v = s.validationErrorPerSample.getOrElse(s.penalizedObjectivePerSample)
    if (math.abs(v) < eps) false
    else {
      if (v > max) {
        max = v
        maxi = c
        true
      } else {
        if (c - maxi >= i) false
        else true
      }
    }
  }

}

case class Step[I](state: I,
                   penalizedObjectivePerSample: Double,
                   validationErrorPerSample: Option[Double],
                   relObj: Double,
                   relValE: Option[Double],
                   count: Int) {
  override def toString =
    s"Step(st=$state,o=$penalizedObjectivePerSample ,v=${validationErrorPerSample
      .getOrElse("NA")},ro=$relObj,rv=${relValE.getOrElse("NA")},c=$count)"
}

trait EvaluateFit[E, @specialized(Double) P] extends Prediction[P] {
  def unpenalizedObjectivePerSample[T: MatOps](batch: Batch[T]): Double
  def evaluateFit[T: MatOps](batch: Batch[T]): E
}

trait ItState {
  def point: Vec[Double]
}

trait Updater[I <: ItState] {
  def next[T: MatOps](x: Vec[Double],
                      b: Batch[T],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[I]): I
}

object SgdResultAggregator extends StrictLogging {

  def aggregateVecs(s: Seq[Vec[Double]]) = {
    Mat(s: _*).rows.map(_.mean2).toVec
  }

  def make[E, P] =
    new stat.crossvalidation.Aggregator[SgdResultWithErrors[E, P]] {
      def aggregate(s: Seq[SgdResultWithErrors[E, P]]) = {
        val estimates = aggregateVecs(s.map(_.result.estimatesV))
        logger.debug("Aggregating {} sgd results. Non zero: {}",
                     s.size,
                     estimates.countif(_ != 0.0))
        SgdResultWithErrors(SgdResult(estimates,
                                      s.head.result.model,
                                      s.head.result.kernel,
                                      s.head.result.normalizer),
                            s.map(_.trainingErrorPerSample).toVec.mean2,
                            None)
      }
    }
}

case class SgdResult[E, P](
    estimatesV: Vec[Double],
    model: ObjectiveFunction[E, P],
    kernel: FeatureMap,
    normalizer: Vec[Double]
) extends Prediction[P]
    with EvaluateFit[E, P] {

  def predict(v: Vec[Double]) = predict(Mat(v).T).raw(0)

  def predict(m: Mat[Double]) =
    model.predictMat(
      estimatesV,
      kernel.applyMat(m.mDiagFromRight(normalizer))(DenseMatOps))

  // sparse matrix does no normalization at the moment TODO
  def predict[T: MatOps](m: T) =
    model.predict(estimatesV, kernel.applyMat(m))

  def unpenalizedObjectivePerSample[T: MatOps](b: Batch[T]): Double =
    model.apply(estimatesV, Batch(b, kernel)) / b.y.length

  def evaluateFit[T: MatOps](b: Batch[T]): E =
    model.eval(estimatesV, Batch(b, kernel))

}

case class NamedSgdResult[E, P](
    raw: SgdResult[E, P],
    names: Index[String]
) extends NamedPrediction[P]
    with Prediction[P] {
  def estimatesV = raw.estimatesV

  def predict(v: Vec[Double]) = raw.predict(v)
  def predict(m: Mat[Double]) = raw.predict(m)
  def predict[T: MatOps](m: T) = raw.predict(m)

}

object NamedSgdResult {
 
  implicit def readwriter[E, P](implicit rw1: ReadWriter[Index[String]]): ReadWriter[NamedSgdResult[E, P]] = {
    implicit val tmp = macroRW[SgdResult[E,P]]
    macroRW[NamedSgdResult[E,P]]
  }
    

  def write(x: NamedSgdResult[_, _]) = upickle.default.write(x)
  def read[E, P](s: String) =
    upickle.default.read[NamedSgdResult[E,P]](s)
}

case class SgdResultWithErrors[E, P](
    result: SgdResult[E, P],
    trainingErrorPerSample: Double,
    validationErrorPerSample: Option[(Double, E)])

object Sgd extends StrictLogging {

  def optimizeFrame[RX: ST: ORD, I <: ItState, E, P](
      data: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_] = L2(0d),
      upd: Updater[I] = CoordinateDescentUpdater(false),
      addIntercept: Boolean = true,
      maxIterations: Int = 100000,
      minEpochs: Double = 1d,
      convergedAverage: Int = 50,
      stop: StoppingCriterion = RelativeStopTraining(1E-6),
      rng: scala.util.Random = new scala.util.Random(42),
      kernel: FeatureMap = IdentityFeatureMap,
      normalize: Boolean = true,
      batchSize: Option[Int] = None
  ) =
    optimize(data,
             yKey,
             obj,
             pen,
             upd,
             addIntercept,
             maxIterations,
             minEpochs,
             convergedAverage,
             stop,
             rng,
             kernel,
             normalize,
             batchSize)

  def optimize[RX: ST: ORD, I <: ItState, E, P](
      f: Frame[RX, String, Double],
      yKey: String,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_] = L2(0d),
      upd: Updater[I] = CoordinateDescentUpdater(false),
      addIntercept: Boolean = true,
      maxIterations: Int = 100000,
      minEpochs: Double = 1d,
      convergedAverage: Int = 50,
      stop: StoppingCriterion = RelativeStopTraining(1E-6),
      rng: scala.util.Random = new scala.util.Random(42),
      kernel: FeatureMap = IdentityFeatureMap,
      normalize: Boolean = true,
      batchSize: Option[Int] = None
  ): Option[NamedSgdResult[E, P]] = {
    val (x, y, std) =
      createDesignMatrix(f, yKey, addIntercept)

    val matrixData = MatrixData(
      trainingX = x,
      trainingY = y,
      penalizationMask = Vec(0d +: std.toSeq.drop(1).map(x => 1d): _*))

    optimize(matrixData,
             obj,
             pen,
             upd,
             kernel,
             maxIterations,
             minEpochs,
             convergedAverage,
             stop,
             batchSize.getOrElse(f.numRows),
             rng,
             normalize,
             None).map { result =>
      val idx =
        obj
          .adaptParameterNames(
            if (addIntercept)
              ("intercept" +: f.colIx.toSeq.filter(_ != yKey))
            else f.colIx.toSeq.filter(_ != yKey))
          .toIndex
      NamedSgdResult(result.result, idx)
    }
  }

  def optimize[D, I <: ItState, E, P, M](
      data: D,
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_],
      upd: Updater[I],
      kernel: FeatureMap,
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      stop: StoppingCriterion,
      batchSize: Int,
      rng: scala.util.Random,
      normalize: Boolean,
      validationBatch: Option[Batch[M]]
  )(implicit dsf: DataSourceFactory[D, M]): Option[SgdResultWithErrors[E, P]] = {
    import dsf.ops

    val (normed, normalizer) = dsf.normalize(data)

    val d1 =
      if (normalize) dsf.apply(normed, None, batchSize, rng)
      else dsf.apply(data, None, batchSize, rng)

    val n1 = if (normalize) normalizer else vec.ones(normalizer.length)

    optimize(d1,
             obj,
             pen,
             upd,
             kernel,
             maxIterations,
             minEpochs,
             convergedAverage,
             stop,
             validationBatch,
             None).map { result =>
      SgdResultWithErrors(
        SgdResult(result.result.estimatesV, result.result.model, kernel, n1),
        result.trainingErrorPerSample,
        result.validationErrorPerSample)
    }
  }

  def optimize[I <: ItState, E, P, M: MatOps](
      dataSource: DataSource[M],
      obj: ObjectiveFunction[E, P],
      pen: Penalty[_],
      updater: Updater[I],
      kernel: FeatureMap,
      maxIterations: Int,
      minEpochs: Double,
      convergedAverage: Int,
      stop: StoppingCriterion,
      validationBatch: Option[Batch[M]],
      start: Option[Vec[Double]]
  ): Option[SgdResultWithErrors[E, P]] = {

    val data: Iterator[Batch[M]] = dataSource.training.flatten

    val firstBatch = Batch(data.next, kernel)

    val validationBatchWithFeatureMap =
      validationBatch.map(b => Batch(b, kernel))

    logger.debug(
      "Call to optimize. obj: {}, pen: {}, updater: {}, maxIter: {}, minEpochs: {}, tail: {}, stop: {}, validation: {}, start: {}",
      obj,
      pen,
      updater,
      maxIterations,
      minEpochs,
      convergedAverage,
      stop,
      validationBatch.isDefined,
      start.isDefined)

    def iteration(
        max: Int,
        min: Int,
        tail: Int): Option[(Vec[Double], Double, Option[(Double, E)])] = {

      val lastIterationSteps: Seq[Step[I]] = {
        // keeping pre with span leaks memory (accumulates all steps)
        val (pre, suf) = iterationStream.takeWhile { x =>
          val nan = x.penalizedObjectivePerSample.isNaN
          if (nan) {
            logger.warn("NaN, stopping iteration. Last state: {}", x)
          }
          !nan
        }.drop(min).take(max).span { x =>
          val cont = stop.continue(x)
          if (!cont) {
            logger.debug("Converged {}", x.count)
          }

          cont
        }

        val tsuf = suf.take(tail).toVector
        val tpre = pre.toVector

        val lastidx = (tpre ++ tsuf).lastOption.map(_.count)
        if (lastidx.isEmpty) {
          logger.warn("Did not converge.")
        }

        if (validationBatch.isEmpty) tpre.takeRight(tail - tsuf.size) ++ tsuf
        else
          (tpre ++ tsuf)
            .sortBy(x => x.validationErrorPerSample.get)
            .takeRight(tail)
      }

      // {
      //
      //   val batch = data.next
      //
      //   import org.nspl._
      //   import org.nspl.awtrenderer._
      //   import org.nspl.saddle._
      //   import org.nspl.data._
      //
      //   def fun(x: Double, y: Double) =
      //     obj.apply(Vec(x, y), batch) + pen
      //       .apply(Vec(x, y), batch.penalizationMask) * (-1)
      //
      //   def fun2(v: Vec[Double]) =
      //     obj.apply(v, batch) + pen.apply(v, batch.penalizationMask) * (-1)
      //
      //   val ct = linesegments(
      //     contour(
      //       list.map(_.get(0)).min - 0.1,
      //       0.1 + list.map(_.get(0)).max,
      //       list.map(_.get(1)).min - 0.1,
      //       0.1 + list.map(_.get(1)).max,
      //       50,
      //       100
      //     )((x, y) => fun(x, y)))
      //
      //   val contourplot = xyplot(
      //     ct,
      //     list
      //       .sliding(2)
      //       .map(x => (x(0).raw(0), x(0).raw(1), x(1).raw(0), x(1).raw(1)))
      //       .toSeq -> lineSegment(),
      //     list.zipWithIndex
      //       .map(x => (x._1.raw(0), x._1.raw(1), x._2.toDouble)) -> point(
      //       size = 2,
      //       color = HeatMapColors(0, list.size))
      //   )(main = updater.toString)
      //
      //   scala.util.Try {
      //     val objplot = xyplot(list.reverse.zipWithIndex.map(x =>
      //       x._2.toDouble -> fun2(x._1)))(axisMargin = 0.05)
      //
      //     show(group(contourplot, objplot, TableLayout(2)))
      //   }
      // }

      if (lastIterationSteps.isEmpty) {
        None
      } else {

        val averagedPoint = lastIterationSteps
            .map(_.state.point)
            .reduce(_ + _) / lastIterationSteps.size

        logger.trace("active in averaged point: {} (n={})",
                     averagedPoint.countif(_ != 0d),
                     lastIterationSteps.size)

        val trainingErrorOfAveragedPoint = lastIterationSteps
            .map(x => x.penalizedObjectivePerSample)
            .reduce(_ + _) / lastIterationSteps.size

        val validationErrorOfAveragedPoint =
          validationBatchWithFeatureMap.map(b =>
            obj.apply(averagedPoint, b) / b.y.length)

        val validationEvalOfAveragedPoint = validationErrorOfAveragedPoint.map(
          x => x -> obj.eval(averagedPoint, validationBatchWithFeatureMap.get))

        validationEvalOfAveragedPoint.foreach(
          validationEvalOfAveragedPoint =>
            logger.debug(
              "Eval on {} samples. Validation - obj: {} / misc: {}. Training obj: {}",
              validationBatchWithFeatureMap.get.y.length,
              validationEvalOfAveragedPoint._1,
              validationEvalOfAveragedPoint._2,
              trainingErrorOfAveragedPoint
          ))

        if (validationEvalOfAveragedPoint.isEmpty) {
          logger.debug(
            "Training obj: {}",
            trainingErrorOfAveragedPoint
          )
        }

        Some(
          (averagedPoint,
           trainingErrorOfAveragedPoint,
           validationEvalOfAveragedPoint))
      }

    }

    def iterationStream: Stream[Step[I]] = {

      def loop(previousStep: Step[I]): Stream[Step[I]] =
        previousStep #:: loop({
          val batch =
            if (firstBatch.full) firstBatch
            else Batch(data.next, kernel)

          val n =
            updater.next(previousStep.state.point,
                         batch,
                         obj,
                         pen,
                         Some(previousStep.state))

          val currentValidationErrorPerSample =
            validationBatchWithFeatureMap.map(b =>
              obj.apply(n.point, b) / b.y.length)

          val currentObjectivePerSample = (obj
              .apply(n.point, batch) - pen.apply(
              n.point,
              obj.adaptPenalizationMask(batch))) / batch.y.length

          val relObj = (currentObjectivePerSample - previousStep.penalizedObjectivePerSample) / currentObjectivePerSample

          val relValE = currentValidationErrorPerSample.map { c =>
            (c - previousStep.validationErrorPerSample.get) / c
          }

          val next =
            Step(state = n,
                 penalizedObjectivePerSample = currentObjectivePerSample,
                 validationErrorPerSample = currentValidationErrorPerSample,
                 relObj = relObj,
                 relValE = relValE,
                 count = previousStep.count + 1)

          logger.trace(
            "{}. Active: {}",
            next,
            n.point.countif(_ != 0d)
          )

          next

        })

      val batch = if (firstBatch.full) firstBatch else Batch(data.next, kernel)
      val start1 = start.getOrElse(obj.start(batch.x.numCols))

      logger.trace(
        "Start state: {}",
        start1
      )

      val firstStep = updater.next(start1, batch, obj, pen, None)

      logger.trace(
        "{}. Active: {}",
        firstStep,
        firstStep.point.countif(_ != 0d)
      )

      val currentValidationErrorPerSample =
        validationBatchWithFeatureMap.map(b =>
          obj.apply(firstStep.point, b) / b.y.length)

      val currentObjectivePerSample = (obj
          .apply(firstStep.point, batch) - pen.apply(
          firstStep.point,
          obj.adaptPenalizationMask(batch))) / batch.y.length

      loop(
        Step(firstStep,
             currentObjectivePerSample,
             currentValidationErrorPerSample,
             1.0,
             Some(1.0),
             1)
      )
    }

    iteration(maxIterations,
              (minEpochs * dataSource.batchPerEpoch).toInt + 1,
              convergedAverage).map {
      case (point, tE, vE) =>
        SgdResultWithErrors(SgdResult(point, obj, kernel, Vec(0d)), tE, vE)
    }

  }

}
