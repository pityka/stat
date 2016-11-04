package stat

import org.saddle._
import scala.util.Random
import stat.sgd.EvaluateFit
import slogging.StrictLogging

package object crossvalidation extends StrictLogging {

  def trainOnTestEvalOnHoldout[E, H](
      idx: Vec[Int],
      trainer: Train[E, H],
      cvmode: CVSplit,
      hyper: H
  ): Iterator[(EvalR[E], Vec[Double])] = {
    cvmode.generate(idx).map {
      case (test, holdout) =>
        logger.debug("train: {} , eval: {} ", test.length, holdout.length)
        val fit = trainer.train(test, hyper)
        (fit.eval(holdout), fit.estimatesV)
    }
  }
  def trainOnTestEvalOnHoldout[E](
      idx: Vec[Int],
      trainer: Train2[E],
      cvmode: CVSplit
  ): Iterator[(EvalR[E], Vec[Double])] = {
    cvmode.generate(idx).map {
      case (test, holdout) =>
        logger.debug("train: {} , eval: {} ", test.length, holdout.length)
        val fit = trainer.train(test)
        (fit.eval(holdout), fit.estimatesV)
    }
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
