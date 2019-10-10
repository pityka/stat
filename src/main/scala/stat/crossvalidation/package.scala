package stat

import org.saddle._
import org.saddle.linalg._
import scala.util.Random
import stat.sgd.EvaluateFit
import slogging.StrictLogging

package object crossvalidation extends StrictLogging {

  // def modifyTrainer[E, H, R](
  //     trainer: Train[E, H, R]
  // )(f: Train[E, H, R] => Train[E, H, R]): Train[E, H, R] = {
  //   val t2 = f(trainer)
  //   new Train[E, H, R] {
  //   def train(idx: Vec[Int], hyper: H): Option[R] = {
  //     t2.train(idx,hyper)
  //   }
  //
  //   def eval(r: R) = t2.eval(r)
  //
  // }

  def bootstrapAggregate[E, H, R, K](
      trainer: Train[E, H, R, K],
      splits: Int,
      aggregator: Aggregator[R],
      rng: Random
  ): Train[E, H, R, K] = new Train[E, H, R, K] {
    def train(idx: Vec[Int],
              hyper: HyperParameter[H, K],
              evalIdx: Option[Vec[Int]],
              start: Option[Vec[Double]]): Option[R] = {
      val bootstrapSamples: Seq[Vec[Int]] = 0 until splits map { _ =>
        0 until idx.length map { j =>
          idx.raw(rng.nextInt(idx.length))
        } toVec
      }
      val bsFits = bootstrapSamples.flatMap { idx =>
        logger.debug("Training bootstrapped sample with sample size {}",
                     idx.length)
        trainer.train(idx, hyper, evalIdx, start)
      }
      if (bsFits.isEmpty) None
      else {
        Some(aggregator.aggregate(bsFits))

      }
    }

    def eval(r: R) = trainer.eval(r)

  }

  def trainOnTestEvalOnHoldout[E, H, R, K](
      idx: Vec[Int],
      trainer: Train[E, H, R, K],
      cvmode: CVSplit,
      hyper: HyperParameter[H, K],
      start: Option[Vec[Double]]
  ): Iterator[(EvalR[E], Vec[Double])] = {
    val splits = cvmode.generate(idx)

    if (splits.isEmpty) Iterator.empty
    else {
      val (test, holdout) = splits.next
      logger.debug("train: {} , eval: {} ", test.length, holdout.length)
      val first = trainer.train(test, hyper, Some(holdout), start).map {
        result =>
          val fit = trainer.eval(result)
          (fit.eval(holdout), fit.estimatesV)
      }

      val rest = splits.map {
        case (test, holdout) =>
          val start = first.map(_._2)
          logger.debug(
            "train: {} , eval: {}, using warm starting estimate from the first split ",
            test.length,
            holdout.length)
          trainer.train(test, hyper, Some(holdout), start).map { result =>
            val fit = trainer.eval(result)
            (fit.eval(holdout), fit.estimatesV)
          }
      }.filter(_.isDefined).map(_.get)

      List(first).iterator.filter(_.isDefined).map(_.get) ++ rest

    }

  }
  def trainOnTestEvalOnHoldout2[E, P, H](
      idx: Vec[Int],
      trainer: Train2[E, P, H],
      cvmode: CVSplit
  ): Iterator[(EvalR[E], Vec[Double], HyperParameter[P, H])] = {
    cvmode
      .generate(idx)
      .map {
        case (test, holdout) =>
          logger.trace("train: {} , eval: {} ", test.length, holdout.length)
          trainer.train(test).map {
            case (fit, hyperp) =>
              (fit.eval(holdout), fit.estimatesV, hyperp)
          }
      }
      .filter(_.isDefined)
      .map(_.get)
  }

  def rSquared(predicted: Vec[Double], y: Vec[Double]) = {
    {
      val residualSS = {
        val r = y - predicted
        r dot r
      }
      val totalSS = {
        val r = y.demeaned
        r dot r
      }
      1.0 - residualSS / totalSS
    }
  }

}
