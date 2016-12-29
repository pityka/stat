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
  ): Option[Double] = {
    val candidates = array.linspace(min, max, n).map(x => math.exp(x))

    val means = candidates.map { c =>
      logger.debug("trainOnTestEvalOnHoldout - hyper: {} - size: {}",
                   c,
                   idx.length)
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec.map(
        _._1.obj)

      val rmean = if (r.count > 1) Some(r.mean) else None

      logger.debug("gridSearch {} : {}", c, rmean)
      (c, rmean)
    }.filter(_._2.isDefined).map(x => x._1 -> x._2.get)

    val optim: Option[Double] =
      if (means.isEmpty) None else Some(means.maxBy(_._2)._1)

    logger.debug("gridSearch. Optim: {}", optim)

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
  ): Option[Double] = {
    val candidates = 0 until n map (i =>
                                      runif() * (max - min) + min) map (math.exp)

    val means = candidates.map { c =>
      logger.debug("trainOnTestEvalOnHoldout - hyper: {} - size: {}",
                   c,
                   idx.length)
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec.map(
        _._1.obj)

      val rmean = if (r.count > 1) Some(r.mean) else None

      logger.debug("randomSearch1D {} : {}", c, rmean)
      (c, rmean)
    }.filter(_._2.isDefined).map(x => x._1 -> x._2.get)

    val optim: Option[Double] =
      if (means.isEmpty) None else Some(means.maxBy(_._2)._1)

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
  ): Option[(Double, Double)] = {
    val candidates = {
      val z1 = 0 until n map (i => runif() * (max - min) + min) map (math.exp)
      val z2 = 0 until n map (i => runif() * (max - min) + min) map (math.exp)
      z1 zip z2
    }

    val means = candidates.map { c =>
      logger.debug("trainOnTestEvalOnHoldout - hyper: {} - size: {}",
                   c,
                   idx.length)
      val r = trainOnTestEvalOnHoldout(idx, trainer, split, c).toSeq.toVec.map(
        _._1.obj)

      val rmean = if (r.count > 1) Some(r.mean) else None

      logger.debug("randomSearch1D {} : {}", c, rmean)
      (c, rmean)
    }.filter(_._2.isDefined).map(x => x._1 -> x._2.get)

    val optim: Option[(Double, Double)] =
      if (means.isEmpty) None else Some(means.maxBy(_._2)._1)

    logger.debug("randomSearch1D. Optim: {}", optim)

    optim
  }
}
