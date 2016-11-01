package stat

import org.saddle._
import scala.util.Random
import stat.sgd.EvaluateFit
import slogging.StrictLogging

package object crossvalidation extends StrictLogging {

  def trainOnTestEvalOnHoldout[E](
      idx: Vec[Int],
      trainer: Train[E],
      cvmode: CVSplit,
      hyper: Double
  ): Iterator[(EvalR[E], Vec[Double])] = {
    cvmode.generate(idx).map {
      case (test, holdout) =>
        logger.debug("train: {} , eval: {} ", test.length, holdout.length)
        val fit = trainer.train(test, hyper)
        (fit.eval(holdout), fit.estimatesV)
    }
  }

  def gridSearch1D[E](
      idx: Vec[Int],
      trainer: Train[E],
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

  def rSquared(predicted: Vec[Double], y: Vec[Double]) = {
    {
      val residualSS = {
        val r = y - predicted
        r dot r
      }
      val totalSS = {
        val r = y - y.mean
        r dot r
      }
      1.0 - residualSS / totalSS
    }
  }

}
