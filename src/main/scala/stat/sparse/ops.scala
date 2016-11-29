package stat.sparse

import org.saddle._
import org.saddle.linalg._
import stat.matops._

object SparseVecOps extends VecOps[SVec] {
  type T = SVec
  def length(t: T): Int = t.length
  def vv(t: T, t2: Vec[Double]): Double = {
    var s = 0d
    var i = 0
    val v = t.values.toVec
    val idx = t.values.index
    while (i < v.length) {
      val vv = v.raw(i)
      val iv = idx.raw(i)
      s += vv * t2.raw(iv)
      i += 1
    }
    s
  }
}

object SparseMatOps extends MatOps[SMat] {
  type V = SVec
  type T = SMat
  implicit val vops: VecOps[V] = SparseVecOps
  def mv(t: T, v: Vec[Double]): Vec[Double] =
    t.map(row => vops.vv(row, v)).toVec

  def tmv(t: T, v: Vec[Double]): Vec[Double] = {
    val sums = Array.ofDim[Double](t.head.length)
    var i = 0
    var j = 0
    while (i < t.size) {
      val row = t(i)
      val vec = row.values.toVec
      val idx = row.values.index
      val other = v.raw(i)
      while (j < vec.length) {
        val vv = vec.raw(j)
        val iv = idx.raw(j)
        sums(iv) += vv * other
        j += 1
      }
      j = 0
      i += 1
    }
    sums
  }

  def innerM(t: T): Mat[Double] = {
    Mat(0 until t.head.length map { colIdx =>
      val r = t.map(r => get(r, colIdx)).toVec

      tmv(t, r)
    }: _*)
  }

  def singularValues(t: T, i: Int): Vec[Double] = {
    val xxt: Mat[Double] = Mat(t.map { rowi =>
      t.map { rowj =>
        rowi.values.joinMap(rowj.values, index.InnerJoin)(_ * _).sum
      }.toVec
    }: _*)
    xxt.eigenValuesSymm(i).map(math.sqrt)

  }

  def mDiagFromLeft(t: T, v: Vec[Double]): T = {
    t.zipWithIndex.map {
      case (row, i) =>
        SVec(row.values * v.raw(i), row.length)
    }
  }

  def tmm(t: T, t2: T): Mat[Double] =
    Mat(0 until t.head.length map { colIdx =>
      val r = t2.map(r => get(r, colIdx)).toVec

      tmv(t, r)
    }: _*)

  def vm(t: SVec, m: Mat[Double], sums: Array[Double], start: Int): Unit = {
    var i = 0
    var j = 0
    val vec = t.values.toVec
    val idx = t.values.index
    while (j < vec.length) {
      val v = vec.raw(j)
      val iv = idx.raw(j)
      while (i < m.numCols) {
        val mat = m.raw(iv, i)
        sums(i + start) += mat * v
        i += 1
      }
      i = 0
      j += 1
    }

  }

  def mm(t: T, m: Mat[Double]): Mat[Double] = {
    val rows = numRows(t)
    val ar = Array.ofDim[Double](m.numCols * rows)
    var i = 0
    var j = 0
    var c = 0
    while (i < t.size) {
      val trow = t(i)
      val prodrow = vm(trow, m, ar, c)
      c += m.numCols
      j = 0
      i += 1
    }
    Mat(rows, m.numCols, ar)
  }
  def numRows(t: T): Int = stat.sparse.numRows(t)
  def numCols(t: T): Int = stat.sparse.numCols(t)
  def row(t: T, i: Int): SVec = t(i)
  def raw(t: T, i: Int, j: Int): Double = get(t(i), j)
  def rows(t: T): IndexedSeq[V] = t
}
