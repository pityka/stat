package stat.crossvalidation

import org.saddle._
import scala.util.Random

case class EvalR[E](obj: Double, misc: E)

trait Train2[EvalRes] {
  def train(idx: Vec[Int]): Option[Eval[EvalRes]]
}

trait Train[EvalRes, Hyper] {
  def train(idx: Vec[Int], hyper: Hyper): Option[Eval[EvalRes]]
}

trait HyperParameterSearch[H] {
  def apply[E](
      idx: Vec[Int],
      t: Train[E, H],
      split: CVSplit,
      min: Double,
      max: Double,
      n: Int
  ): Option[H]
}

object Train {
  def nestedSearch[E, H](t: Train[E, H],
                         split: CVSplit,
                         min: Double,
                         max: Double,
                         n: Int,
                         search: HyperParameterSearch[H]) = new Train2[E] {
    def train(idx: Vec[Int]): Option[Eval[E]] = {
      search(
        idx,
        t,
        split,
        min,
        max,
        n
      ).flatMap { opt =>
        t.train(idx, opt)
      }
    }
  }
}

trait Eval[EvalRes] {
  def eval(idx: Vec[Int]): EvalR[EvalRes]
  def estimatesV: Vec[Double]
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

case class KFold(folds: Int, rng: Random, replica: Int) extends CVSplit {
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
