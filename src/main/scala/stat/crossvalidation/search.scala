package stat.crossvalidation

import org.saddle._
import scala.util.Random
import stat.sgd.EvaluateFit
import slogging.StrictLogging

trait Generator[T] {
  def gen(n: Int): Seq[T]
}

case class DoubleGrid(min: Double, max: Double) extends Generator[Double] {
  def gen(n: Int) = array.linspace(min, max, n).map(x => math.exp(x)).toSeq
}

case class DoubleRandom(min: Double, max: Double)(runif: () => Double)
    extends Generator[Double] {
  def gen(n: Int) =
    0 until n map (i => runif() * (max - min) + min) map (math.exp)
}

case class CompositeGenerator2[T1, T2, T](
    g1: Generator[T1],
    g2: Generator[T2])(f: ((T1, T2)) => T)
    extends Generator[T] {
  def gen(n: Int) = (g1.gen(n).zip(g2.gen(n))).map(f)
}

trait HyperParameterSearch[H, K] {
  def apply[E, R](
      idx: Vec[Int],
      t: Train[E, H, R, K],
      split: CVSplit
  ): Option[(HyperParameter[H, K], Option[Vec[Double]])]
}

object HyperParameterSearch {

  def Random2DGenerator(min1: Double,
                        max1: Double,
                        min2: Double,
                        max2: Double)(runif: () => Double) =
    CompositeGenerator2(DoubleRandom(min1, max1)(runif),
                        DoubleRandom(min2, max2)(runif))(x => x)

  def GridSearch1D(min: Double, max: Double, n: Int) =
    apply(DoubleGrid(min, max), DoubleGrid(0d, 0d), n)

  def RandomSearch2D(min1: Double,
                     max1: Double,
                     min2: Double,
                     max2: Double,
                     n: Int)(runif: () => Double) =
    apply(Random2DGenerator(min1, max1, min2, max2)(runif),
          DoubleGrid(0d, 0d),
          n)

  def GridSearch(min: Double, max: Double, n: Int) =
    apply(DoubleGrid(min, max), DoubleGrid(0d, 0d), n)

  def apply[PH, KH](penalty: Generator[PH], kernel: Generator[KH], n: Int) =
    new HyperParameterSearch[PH, KH] with StrictLogging {
      def apply[E, R](
          idx: Vec[Int],
          trainer: Train[E, PH, R, KH],
          split: CVSplit
      ): Option[(HyperParameter[PH, KH], Option[Vec[Double]])] = {

        val candidates = penalty.gen(n) zip kernel.gen(n) map (x =>
                                                                 HyperParameter(
                                                                   x._1,
                                                                   x._2))

        logger.debug("Warming up: hyper - {}", candidates.head)
        val start: Option[Vec[Double]] =
          trainer.train(idx, candidates.head, None, None).map { result =>
            trainer.eval(result).estimatesV
          }

        val means = {

          candidates.map { c =>
            logger.debug("trainOnTestEvalOnHoldout - hyper: {} - size: {}",
                         c,
                         idx.length)
            val r =
              trainOnTestEvalOnHoldout(idx, trainer, split, c, start).toSeq.toVec
                .map(_._1.obj)

            val rmean = if (r.count > 1) Some(r.mean) else None

            logger.debug("HyperParameterSearch. Hyper: {} : mean: {}",
                         c,
                         rmean)
            (c, rmean)
          }.filter(_._2.isDefined).map(x => x._1 -> x._2.get)
        }

        val optim =
          if (means.isEmpty) None else Some((means.maxBy(_._2)._1, start))

        logger.debug("HyperParameterSearch. Chosen hyperparameter: {}", optim)

        optim
      }
    }
}
