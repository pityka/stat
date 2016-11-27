package stat

import org.saddle._

package object matops {
  implicit class Pimp[T](t: T)(implicit val op: MatOps[T]) {
    // type V = op.V
    def mv(v: Vec[Double]): Vec[Double] = op.mv(t, v)
    def tmv(v: Vec[Double]): Vec[Double] = op.tmv(t, v)
    def innerM: Mat[Double] = op.innerM(t)
    def singularValues(i: Int): Vec[Double] = op.singularValues(t, i)
    def mDiagFromLeft(v: Vec[Double]): T = op.mDiagFromLeft(t, v)
    def tmm(t2: T): Mat[Double] = op.tmm(t, t2)
    def mm(m: Mat[Double]): Mat[Double] = op.mm(t, m)

    def numRows: Int = op.numRows(t)
    def numCols: Int = op.numCols(t)
    def row(i: Int): Vec[Double] = op.row(t, i)
    def raw(i: Int, j: Int): Double = op.raw(t, i, j)

    // implicit def vops: VecOps[V] = op.vops
  }
  // object Pimp {
  //   type Aux[A, B] = Pimp[A] { type V = B }
  // }

  implicit class PimpV[T](t: T)(implicit val op: VecOps[T]) {
    def length: Int = op.length(t)
    def dot(t2: Vec[Double]): Double = op.dot(t, t2)
  }

  implicit def vops[T](implicit m: MatOps[T]): VecOps[m.V] = m.vops
  // implicit val dense: MatOps[Mat[Double]] = DenseMatOps
}
