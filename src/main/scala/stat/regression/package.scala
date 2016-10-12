package stat

import org.saddle._

package object regression {

  def createDesignMatrix[I](
      data: Frame[I, String, Double],
      missingMode: MissingMode,
      intercept: Boolean
  )(implicit ev: org.saddle.ST[I],
    ord: Ordering[I]): Frame[I, String, Double] =
    missing(if (intercept)
              addIntercept(data)
            else data,
            missingMode)

  def addIntercept[I: ST: Ordering](f: Frame[I, String, Double]) =
    Frame(vec.ones(f.numRows) +: f.toColSeq.map(_._2.toVec),
          f.rowIx,
          Index("intercept").concat(f.colIx))

  /**
    * Creates a completely filled in data matrix with intercept column.
    */
  def missing[I](
      data: Frame[I, String, Double],
      missingMode: MissingMode
  )(implicit ev: org.saddle.ST[I],
    ord: Ordering[I]): Frame[I, String, Double] =
    missingMode match {
      case MeanImpute =>
        data.mapVec(column => meanImpute(column))
      case DropSample =>
        data.rfilter(row => !(row.hasNA || row.toVec.exists(_.isInfinite)))
    }

  def meanImpute(data: Vec[Double]): Vec[Double] = {
    val mean = data.mean
    data.fillNA(_ => mean)
  }

}
