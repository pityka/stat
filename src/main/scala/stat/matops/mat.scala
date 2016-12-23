package stat.matops

import org.saddle._

trait VecOps[T] {
  def length(t: T): Int
  // dot
  def vv(t: T, t2: Vec[Double]): Double
  // dot
  def vv2(t: T, t2: T): Double
}

trait LinearMap[T] {
  def mv(t: T, v: Vec[Double]): Vec[Double]
  def numCols(t: T): Int
}

object LinearMap {
  implicit class Pimp[T](t: T)(implicit val op: LinearMap[T]) {
    def mv(v: Vec[Double]): Vec[Double] = op.mv(t, v)
    def numCols: Int = op.numCols(t)

  }
}

trait MatOps[T] extends LinearMap[T] {
  type V
  implicit val vops: VecOps[V]
  def mv(t: T, v: Vec[Double]): Vec[Double]
  def tmv(t: T, v: Vec[Double]): Vec[Double]
  def innerM(t: T): Mat[Double]
  def singularValues(t: T, i: Int): Vec[Double]
  def mDiagFromLeft(t: T, v: Vec[Double]): T
  def tmm(t: T, t2: T): Mat[Double]
  def mm(t: T, m: Mat[Double]): Mat[Double]
  def numRows(t: T): Int
  def numCols(t: T): Int
  def row(t: T, i: Int): V
  def raw(t: T, i: Int, j: Int): Double
  def rows(t: T): IndexedSeq[V]
}

object Test {

  def test[T: MatOps](a: T, v: Vec[Double]) = {
    val b: Vec[Double] = a mv v
  }

}
