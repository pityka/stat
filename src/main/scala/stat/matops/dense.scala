package stat.matops

import org.saddle._

object DenseVecOps extends VecOps[Vec[Double]] {
  type T = Vec[Double]
  import org.saddle.linalg._
  def length(t: T): Int = t.length
  def vv(t: T, t2: T): Double = t vv t2
  def vv2(t: T, t2: T): Double = vv(t, t2)
}

object DenseMatOps extends MatOps[Mat[Double]] with LinearMap[Mat[Double]] {
  import org.saddle.linalg._
  type T = Mat[Double]
  type V = Vec[Double]
  implicit val vops: VecOps[V] = DenseVecOps
  def mv(t: T, v: Vec[Double]): Vec[Double] = t.mv(v)
  def tmv(t: T, v: Vec[Double]): Vec[Double] = t.tmv(v)
  def innerM(t: T): Mat[Double] = t.innerM
  def outerM(t: T): Mat[Double] = t.outerM
  def singularValues(t: T, i: Int): Vec[Double] = t.singularValues(i)
  def mDiagFromLeft(t: T, v: Vec[Double]): T = t.mDiagFromLeft(v)
  def tmm(t: T, t2: T): Mat[Double] = t.tmm(t2)
  def mm(t: T, m: Mat[Double]): Mat[Double] = t.mm(m)
  def mmLeft(left: Mat[Double], right: Mat[Double]): Mat[Double] =
    left mm right
  def numRows(t: T): Int = t.numRows
  def numCols(t: T): Int = t.numCols
  def row(t: T, i: Int): Vec[Double] = t.row(i)
  def col(t: T, i: Int): Vec[Double] = t.col(i)
  def raw(t: T, i: Int, j: Int): Double = t.raw(i, j)
  def rows(t: T): IndexedSeq[V] = t.rows
  def svd(t: T, i: Int): SVDResult = t.svd(i)
}
