package stat

import org.saddle._
import org.saddle.linalg._

package object pca {

  def fromData[RX: ORD: ST, CX: ORD: ST](
      data: Frame[RX, CX, Double]
  ): PCA[RX, CX] = {
    val SVDResult(u, sigma, vt) = data.toMat.svd

    val proj: Frame[RX, Int, Double] = {
      val projectedM: Mat[Double] = Mat(
        u.cols.zip(sigma.toSeq).map(x => x._1 * x._2): _*)
      Frame(projectedM, data.rowIx, index.IndexIntRange(data.colIx.length))
    }

    val load: Frame[CX, Int, Double] =
      Frame(vt.T, data.colIx, index.IndexIntRange(data.colIx.length))

    PCA(sigma, proj, load)

  }

}
