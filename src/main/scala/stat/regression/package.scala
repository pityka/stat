package stat

import org.saddle._

package object regression {

  def createDesignMatrix[I](
      data: Frame[I, String, Double],
      intercept: Boolean
  )(implicit ev: org.saddle.ST[I],
    ord: Ordering[I]): Frame[I, String, Double] =
    if (intercept)
      addIntercept(data)
    else data

  def addIntercept[I: ST: Ordering](f: Frame[I, String, Double]) =
    Frame(vec.ones(f.numRows) +: f.toColSeq.map(_._2.toVec),
          f.rowIx,
          Index("intercept").concat(f.colIx))

}
