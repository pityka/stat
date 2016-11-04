package stat.crossvalidation

import org.saddle._
import scala.util.Random
import stat.sgd.EvaluateFit
import slogging.StrictLogging

case object GridSearch
    extends HyperParameterSearch[Double]
    with StrictLogging {
  def apply[E](
      idx: Vec[Int],
      trainer: Train[E, Double],
      split: CVSplit,
      min: Double,
      max: Double,
      n: Int
  ): Double = {
    val candidates = array.linspace(min, max, n).map(x => math.exp(x))

    val optim = candidates.maxBy { c =>
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec
        .map(_._1.obj)
        .mean
      logger.debug("gridSearch1D {} : {}", c, r)
      r
    }

    logger.debug("gridSearch1D. Optim: {}", optim)

    optim
  }
}

case class RandomSearch(runif: () => Double)
    extends HyperParameterSearch[Double]
    with StrictLogging {
  def apply[E](
      idx: Vec[Int],
      trainer: Train[E, Double],
      split: CVSplit,
      min: Double,
      max: Double,
      n: Int
  ): Double = {
    val candidates = 0 until n map (i => runif() * (max - min) + min)

    val optim = candidates.maxBy { c =>
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec
        .map(_._1.obj)
        .mean
      logger.debug("randomSearch1D {} : {}", c, r)
      r
    }

    logger.debug("randomSearch1D. Optim: {}", optim)

    optim
  }
}

case class RandomSearch2D(runif: () => Double)
    extends HyperParameterSearch[(Double, Double)]
    with StrictLogging {
  def apply[E](
      idx: Vec[Int],
      trainer: Train[E, (Double, Double)],
      split: CVSplit,
      min: Double,
      max: Double,
      n: Int
  ): (Double, Double) = {
    val candidates = {
      val z1 = 0 until n map (i => runif() * (max - min) + min)
      val z2 = 0 until n map (i => runif() * (max - min) + min)
      z1 zip z2
    }

    val optim = candidates.maxBy { c =>
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec
        .map(_._1.obj)
        .mean
      logger.debug("randomSearch2D {} : {}", c, r)
      r
    }

    logger.debug("randomSearch1D. Optim: {}", optim)

    optim
  }
}
