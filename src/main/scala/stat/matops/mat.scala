package stat.matops

import org.saddle._

trait VecOps[T] {
  def length(t: T): Int
  def vv(t: T, t2: Vec[Double]): Double
}

trait MatOps[T] {
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

object DenseVecOps extends VecOps[Vec[Double]] {
  type T = Vec[Double]
  import org.saddle.linalg._
  def length(t: T): Int = t.length
  def vv(t: T, t2: T): Double = t vv t2
}

object DenseMatOps extends MatOps[Mat[Double]] {
  import org.saddle.linalg._
  type T = Mat[Double]
  type V = Vec[Double]
  implicit val vops: VecOps[V] = DenseVecOps
  def mv(t: T, v: Vec[Double]): Vec[Double] = t.mv(v)
  def tmv(t: T, v: Vec[Double]): Vec[Double] = t.tmv(v)
  def innerM(t: T): Mat[Double] = t.innerM
  def singularValues(t: T, i: Int): Vec[Double] = t.singularValues(i)
  def mDiagFromLeft(t: T, v: Vec[Double]): T = t.mDiagFromLeft(v)
  def tmm(t: T, t2: T): Mat[Double] = t.tmm(t2)
  def mm(t: T, m: Mat[Double]): Mat[Double] = t.mm(m)
  def numRows(t: T): Int = t.numRows
  def numCols(t: T): Int = t.numCols
  def row(t: T, i: Int): Vec[Double] = t.row(i)
  def raw(t: T, i: Int, j: Int): Double = t.raw(i, j)
  def rows(t: T): IndexedSeq[V] = t.rows
}

object Test {

  def test[T: MatOps](a: T, v: Vec[Double]) = {
    val b: Vec[Double] = a mv v
  }

}
