package stat.crossvalidation

import org.saddle._
import scala.util.Random

case class EvalR[E](obj: Double, misc: E)

trait Train[EvalRes] {
  def train(idx: Vec[Int], hyper: Double): Eval[EvalRes]
}

object Train {
  def nestGridSearch[E](t: Train[E],
                        split: CVSplit,
                        min: Double,
                        max: Double,
                        n: Int) = new Train[E] {
    def train(idx: Vec[Int], hyper: Double): Eval[E] = {
      val opt = gridSearch1D(
        idx,
        t,
        split,
        min,
        max,
        n
      )
      t.train(idx, opt)
    }
  }
}

trait Eval[EvalRes] {
  def eval(idx: Vec[Int]): EvalR[EvalRes]
  def estimatesV: Vec[Double]
}

sealed trait CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])]
  def withSeed(s: Int): CVSplit
}

case class Split(ratioOfTrain: Double, seed: Int) extends CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])] = {
    val rng = new Random(seed)
    val indices = (0 until d.length).toVector
    val (in, out) =
      rng.shuffle(indices).splitAt((d.length * ratioOfTrain).toInt)
    List((d.take(in.toArray), d.take(out.toArray))).iterator

  }
  def withSeed(i: Int) = Split(ratioOfTrain, i)
}

case class KFold(folds: Int, seed: Int, replica: Int) extends CVSplit {
  def generate(d: Vec[Int]): Iterator[(Vec[Int], Vec[Int])] = {
    val rng = new Random(seed)
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
  def withSeed(i: Int) = KFold(folds, i, replica)
}

case class KFoldStratified(folds: Int,
                           seed: Int,
                           replica: Int,
                           strata: Seq[Vec[Int]])
    extends CVSplit {
  val kf = KFold(folds, seed, replica)
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
  def withSeed(i: Int) = KFoldStratified(folds, i, replica, strata)
}
