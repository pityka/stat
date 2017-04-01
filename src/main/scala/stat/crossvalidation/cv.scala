package stat.crossvalidation

import org.saddle._
import scala.util.Random

case class EvalR[E](unpenalizedObjectivePerSample: Double, misc: E)

trait Train2[EvalRes, P, K] {
  def train(idx: Vec[Int]): Option[(Eval[EvalRes], HyperParameter[P, K])]
}

trait Train[EvalRes, PHyper, Res, KHyper] {
  def train(idx: Vec[Int],
            hyper: HyperParameter[PHyper, KHyper],
            evalIdx: Option[Vec[Int]],
            start: Option[Vec[Double]]): Option[Res]
  def eval(r: Res): Eval[EvalRes]
}

object Train {
  def nestedSearch[E, H, R, K](t: Train[E, H, R, K],
                               split: CVSplit,
                               search: HyperParameterSearch[H, K],
                               warmStart: Boolean) =
    new Train2[E, H, K] with slogging.StrictLogging {
      def train(idx: Vec[Int]): Option[(Eval[E], HyperParameter[H, K])] = {
        search(
          idx,
          t,
          split,
          warmStart
        ).flatMap {
          case (opt, start) =>
            logger.debug(
              "Retraining with optimal hyperparameter: {}, not using warm start. Samples: {}",
              opt,
              idx.length)
            t.train(idx, opt, None, None).map(r => (t.eval(r), opt))
        }
      }
    }
}

trait Eval[EvalRes] {
  def eval(idx: Vec[Int]): EvalR[EvalRes]
  def estimatesV: Vec[Double]
}

trait Aggregator[R] {
  def aggregate(s: Seq[R]): R
}

sealed trait CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])]
}

case class Split(ratioOfTrain: Double, rng: Random) extends CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])] = {
    val indices = (0 until d.length).toVector
    val (in, out) =
      rng.shuffle(indices).splitAt((d.length * ratioOfTrain).toInt)
    List((d.take(in.toArray), d.take(out.toArray))).iterator

  }
}

case class SplitTakeIndex(idx: Array[Int]) extends CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])] = {
    List(d.take(idx) -> d.without(idx)).iterator
  }
}

case class KFold(folds: Int, rng: Random, replica: Int = 1) extends CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])] = {
    val indices = (0 until d.length).toVector
    (1 to replica iterator) flatMap { r =>
      val shuffled = rng.shuffle(indices)
      shuffled.grouped(shuffled.size / folds + 1).map { ii =>
        val i = ii.toArray
        val out = d.take(i)
        val in = d.without(i)
        (in, out)
      }

    }
  }
}

case class KFoldStratified(folds: Int,
                           rng: Random,
                           replica: Int,
                           strata: Seq[Vec[Int]])
    extends CVSplit {
  val kf = KFold(folds, rng, replica)
  def generate(d: Vec[Int]) = {
    val strataIters = strata.map(s => kf.generate(s))
    new Iterator[(Vec[Int], Vec[Int])] {
      def hasNext = strataIters.head.hasNext
      def next =
        strataIters
          .map(_.next)
          .reduce((x, y) => x._1.concat(y._1) -> x._2.concat(y._2))
    }
  }
}

case class HyperParameter[H, KH](penalty: H, kernel: KH)
