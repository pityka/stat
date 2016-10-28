package stat

import org.saddle._
import org.nspl._
import org.nspl.saddle._
import org.nspl.data._
import org.nspl.awtrenderer._

package object kmedoid {

  def plot(data: Mat[Double], assignment: Vec[Int], max: Int) = {

    val colors = {
      val m = assignment.toSeq.distinct.zipWithIndex.toMap
      assignment.map(m).map(_.toDouble)
    }

    val projections = 0 until max combinations (2) map { g =>
      val c1 = g(0)
      val c2 = g(1)
      val col1 = data.col(c1)
      val col2 = data.col(c2)
      xyplot(
        Mat(col1, col2, colors) -> point(
          labelText = false,
          color = DiscreteColors(assignment.toSeq.distinct.size)))()
    }

    sequence(projections.toList, TableLayout(4))

  }

  def assign(dist: Mat[Double], i: Int, medoids: Vec[Int]): (Int, Double) =
    medoids.toSeq.map(m => m -> dist.raw(i, m)).minBy(_._2)

  def assignAll(dist: Mat[Double], medoids: Vec[Int]): (Vec[Int], Double) = {
    val (assignments, costs) =
      (0 until dist.numRows).map(i => assign(dist, i, medoids)).unzip
    assignments.toVec -> costs.sum
  }

  def cost(dist: Mat[Double], m: Int, cluster: Vec[Int]): Double =
    cluster.foldLeft(0d)((s, i) => s + dist.raw(i, m))

  def recenter(dist: Mat[Double], cluster: Vec[Int], m: Int): Int =
    cluster.toSeq.minBy(i => cost(dist, i, cluster))

  def recenterAll(dist: Mat[Double], assignments: Vec[Int]): Vec[Int] = {
    val assignmentWithIndex = Series(assignments)
    assignments.toSeq.distinct.map { m =>
      val cluster = assignmentWithIndex.filter(_ == m).index.toVec
      recenter(dist, cluster, m)
    }.toVec
  }

  def step(dist: Mat[Double], assignments: Vec[Int]): (Vec[Int], Double) = {
    val newmedoids = recenterAll(dist, assignments)
    val (newassignment, cost) = assignAll(dist, newmedoids)
    (newassignment, cost)
  }

  def apply(dist: Mat[Double], init: Vec[Int]): KMedoidResult = {
    val (_s2, _c) = step(dist, assignAll(dist, init)._1)
    var s2 = _s2
    var c = _c
    var b = true
    while (b) {
      val (_s2, c2) = step(dist, s2)
      s2 = _s2
      b = c2 < c
      c = c2

    }

    KMedoidResult(s2)
  }

}
