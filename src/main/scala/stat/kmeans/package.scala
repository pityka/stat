package stat

import org.saddle._
import org.saddle.linalg._

import org.nspl._
import org.nspl.saddle._
import org.nspl.data._

package object kmeans {

  def plot(data: Mat[Double], res: KMeansResult, max: Int) = {

    val projections = 0 until max combinations (2) map { g =>
      val c1 = g(0)
      val c2 = g(1)
      xyplot(
        Mat((data.col(c1, c2).cols :+ (res.clusters.map(_.toDouble))): _*) -> point(
          labelText = false,
          color = DiscreteColors(res.means.numRows)))()
    }

    sequence(projections.toList, TableLayout(4))

  }

  def assign(v: Vec[Double], means: Mat[Double]): Int = {
    0 until means.numRows minBy { i =>
      v dot means.row(i)
    }
  }

  def assignAll(data: Mat[Double], means: Mat[Double]): Seq[Vec[Int]] =
    data.rows.zipWithIndex.map {
      case (row, idx) =>
        val membership = assign(row, means)
        membership -> idx
    }.groupBy(_._1)
      .toSeq
      .map(x => x._1 -> Vec(x._2.map(_._2): _*))
      .sortBy(_._1)
      .map(_._2)

  def update(data: Mat[Double], memberships: Seq[Vec[Int]]): Mat[Double] = {
    Mat(memberships.map { idx =>
      Vec(data.takeRows(idx: Array[Int]).cols.map(c => c.sum / c.length): _*)
    }: _*).T
  }

  def step(data: Mat[Double], means: Mat[Double]) = {
    val assignment = assignAll(data, means)
    (update(data, assignment), assignment)

  }

  def apply(data: Mat[Double], init: Mat[Double], it: Int): KMeansResult = {
    val (next, assignments) = step(data, init)

    if (it == 0)
      KMeansResult(assignments.zipWithIndex
                     .flatMap(x => x._1.toSeq.map(y => y -> x._2))
                     .sortBy(_._1)
                     .map(_._2)
                     .toVec,
                   next)
    else apply(data, next, it - 1)
  }

}
