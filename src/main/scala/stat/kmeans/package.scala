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
      val col1 = data.map(s => s.values.first(c1).getOrElse(0d)).toVec
      val col2 = data.map(s => s.values.first(c2).getOrElse(0d)).toVec
      xyplot(
        Mat(col1, col2, res.clusters.map(_.toDouble)) -> point(
          labelText = false,
          color = DiscreteColors(res.means.size)))()
    }

    sequence(projections.toList, TableLayout(4))

  }

  def euclid(t1: SVec, t2: SVec) = {
    assert(t1.length == t2.length)
    var s = 0d
    var i = 0
    val v1 = t1.values.toVec
    val v2 = t2.values.toVec
    val idx1 = t1.values.index
    val idx2 = t2.values.index
    val n = v1.length
    while (i < n) {
      val vv1 = if (idx1.contains(i)) v1.raw(idx1.getFirst(i)) else 0d
      val vv2 = if (idx2.contains(i)) v2.raw(idx2.getFirst(i)) else 0d
      val d = vv1 - vv2
      s += d * d

      i += 1
    }
    math.sqrt(s)
  }

  def assign(v: SVec, means: SMat): Int = {
    0 until means.size minBy { i =>
      euclid(v, means(i))
    }
  }

  def colmeans(t: SMat): SVec = {
    val means = stat.sparse.colmeans(t)
    SVec(Series(means).filter(_ != 0d), means.length)
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
    apply(data.rows.map(x => SVec(Series(x), x.length)),
          init.rows.map(x => SVec(Series(x), x.length)),
          it)

  def matToSparse(data: Mat[Double]) =
    data.rows.map(x => SVec(Series(x), x.length))

}
