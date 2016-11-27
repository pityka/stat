package stat.sparse

import org.saddle._
import org.saddle.linalg._

object SparseVecOps extends VecOps[SVec] {
  type T = SVec
  def length(t: T): Int
  def dot(t: T, t2: Vec[Double]): Double
}

// trait MatOps[T] {
//   type V
//   implicit val vops: VecOps[V]
//   def mv(t: T, v: Vec[Double]): Vec[Double]
//   def tmv(t: T, v: Vec[Double]): Vec[Double]
//   def innerM(t: T): Mat[Double]
//   def singularValues(t: T, i: Int): Vec[Double]
//   def mDiagFromLeft(t: T, v: Vec[Double]): T
//   def tmm(t: T, t2: T): Mat[Double]
//   def mm(t: T, m: Mat[Double]): Mat[Double]
//   def numRows(t: T): Int
//   def numCols(t: T): Int
//   def row(t: T, i: Int): Vec[Double]
//   def raw(t: T, i: Int, j: Int): Double
//   def rows(t: T): IndexedSeq[V]
// }
