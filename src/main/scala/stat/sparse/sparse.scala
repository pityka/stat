package stat
import org.saddle._
package object sparse {
  type SMat = IndexedSeq[SVec]

  def length(sv: SVec): Int = sv.length
  def idx(sv: SVec): Vec[Int] = index.IndexIntRange(length(sv)).toVec
  def dense(sv: SVec): Vec[Double] = dense(sv, idx(sv))
  def dense(sv: SVec, idx: Vec[Int]): Vec[Double] =
    idx.map(i => sv.values.first(i).getOrElse(0d))

  def get(sv: SVec, idx: Int) =
    if (idx >= sv.length || idx < 0)
      throw new RuntimeException("index otu of range")
    else sv.values.first(idx).getOrElse(0d)

  def numCols(sm: SMat): Int =
    if (sm.isEmpty) 0 else sm.map(x => length(x)).max
  def numRows(sm: SMat): Int = sm.length

  def dense(sm: SMat, rIdx: Vec[Int], cIdx: Vec[Int]): Mat[Double] =
    Mat(rIdx.map(i => sm(i)).map(dense(_, cIdx)).toSeq: _*).T
  def rowIx(sm: SMat): Vec[Int] = index.IndexIntRange(numRows(sm)).toVec
  def colIx(sm: SMat): Vec[Int] = index.IndexIntRange(numCols(sm)).toVec
  def dense(sm: SMat): Mat[Double] = dense(sm, rowIx(sm), colIx(sm))

}
