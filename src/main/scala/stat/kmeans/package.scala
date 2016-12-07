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
          color = DiscreteColors(res.clusters.length)))()
    }

    sequence(projections.toList, TableLayout(4))

  }

  def euclid(t1: SVec, t2: Vec[Double]) = {
    assert(t1.length == t2.length)
    var s = 0d
    var i = 0
    val v1 = t1.values.toVec
    val idx1 = t1.values.index
    val n1 = v1.length
    while (i < n1) {
      val vv1 = v1.raw(i)
      val iv1 = idx1.raw(i)
      val vv2 = t2.raw(iv1)
      val d = vv1 - vv2
      s += d * d
      i += 1
    }
    i = 0
    math.sqrt(s)
  }

  def assign(v: SVec, means: Vector[Vec[Double]]): Int = {
    var i = 1
    var m = 0
    var mv = euclid(v, means.head)
    while (i < means.size) {
      val f = euclid(v, means(i))
      if (f < mv) {
        mv = f
        m = i
      }
      i += 1
    }
    m
  }

  def colmeans(t: SMat): Vec[Double] =
    stat.sparse.colmeans(t)

  def assignAll(data: SMat, means: Vector[Vec[Double]]): Vector[Vec[Int]] =
    data.zipWithIndex.map {
      case (row, idx) =>
        val membership = assign(row, means)
        membership -> idx
    }.groupBy(_._1)
      .toVector
      .map(x => x._1 -> Vec(x._2.map(_._2): _*))
      .sortBy(_._1)
      .map(_._2)

  def update(data: SMat, memberships: Vector[Vec[Int]]): Vector[Vec[Double]] =
    memberships.map { idx =>
      colmeans(idx.map(i => data(i)).toSeq.toVector)
    }

  def step(data: SMat, means: Vector[Vec[Double]]) = {
    val assignment = assignAll(data, means)
    (update(data, assignment), assignment)
  }

  def apply(data: SMat, init: Vector[Vec[Double]], it: Int): KMeansResult = {
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
          init.rows.toVector,
          it)

  def matToSparse(data: Mat[Double]) =
    data.rows.map(x => SVec(Series(x), x.length))

}
