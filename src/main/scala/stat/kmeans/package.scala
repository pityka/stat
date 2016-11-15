package stat

import org.saddle._
import org.saddle.linalg._

import org.nspl._
import org.nspl.saddle._
import org.nspl.data._
import stat.sparse._

package object kmeans {

  def plot(data: SMat, res: KMeansResult, max: Int) = {

    val projections = 0 until max combinations (2) map { g =>
      val c1 = g(0)
      val c2 = g(1)
      val col1 = data.map(s => s.first(c1).getOrElse(0d)).toVec
      val col2 = data.map(s => s.first(c2).getOrElse(0d)).toVec
      xyplot(
        Mat(col1, col2, res.clusters.map(_.toDouble)) -> point(
          labelText = false,
          color = DiscreteColors(res.means.size)))()
    }

    sequence(projections.toList, TableLayout(4))

  }

  def euclid(v1: SVec, v2: SVec) = {
    val (j1, j2) = v1.align(v2, index.OuterJoin)
    val d = (j1.fillNA(_ => 0d) - j2.fillNA(_ => 0d)).toVec
    math.sqrt(d dot d)
  }

  def assign(v: SVec, means: SMat): Int = {
    0 until means.size minBy { i =>
      euclid(v, means(i))
    }
  }

  def colmeans(data: SMat): SVec = {
    val keys = data.flatMap(_.index.toSeq).distinct
    keys.map { k =>
      k -> data.map(s => s.first(k).getOrElse(0d)).toVec.mean
    }.toSeries
  }

  def assignAll(data: SMat, means: SMat): Seq[Vec[Int]] =
    data.zipWithIndex.map {
      case (row, idx) =>
        val membership = assign(row, means)
        membership -> idx
    }.groupBy(_._1)
      .toSeq
      .map(x => x._1 -> Vec(x._2.map(_._2): _*))
      .sortBy(_._1)
      .map(_._2)

  def update(data: SMat, memberships: Seq[Vec[Int]]): SMat = {
    memberships.map { idx =>
      colmeans(idx.map(i => data(i)).toSeq.toVector)
    }.toVector
  }

  def step(data: SMat, means: SMat) = {
    val assignment = assignAll(data, means)
    (update(data, assignment), assignment)
  }

  def apply(data: SMat, init: SMat, it: Int): KMeansResult = {
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

  def apply(data: Mat[Double], init: Mat[Double], it: Int): KMeansResult =
    apply(data.rows.map(x => Series(x)), init.rows.map(x => Series(x)), it)

  def matToSparse(data: Mat[Double]) = data.rows.map(x => Series(x))

}
