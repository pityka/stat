package stat

import org.saddle._
import org.saddle.linalg._

package object pca {

  def fromData[RX: ORD: ST, CX: ORD: ST](
      data: Frame[RX, CX, Double],
      max: Int
  ): PCA[RX, CX] = {
    val SVDResult(u, sigma, vt) = data.demeaned.toMat.svd(max)

    val sigmaPos = sigma.filter(_ > 1E-4)

    val proj: Frame[RX, Int, Double] = {
      val projectedM: Mat[Double] = Mat(
        u.cols.zip(sigma.toSeq).filter(_._2 > 1E-4).map(x => x._1 * x._2): _*)
      Frame(projectedM, data.rowIx, index.IndexIntRange(sigmaPos.length))
    }

    val load: Frame[CX, Int, Double] =
      Frame(vt.T.takeCols(0 until sigmaPos.length: _*),
            data.colIx,
            index.IndexIntRange(sigmaPos.length))

    PCA(sigmaPos, proj, Some(load))

  }

  def fromCovariance[RX: ORD: ST](
      cov: Frame[RX, RX, Double],
      max: Int
  ) = {
    val EigenDecompositionSymmetric(u, sigma2) = cov.toMat.eigSymm(max)
    val sigma = sigma2.map(math.sqrt)
    val sigmaPos = sigma.filter(_ > 1E-4)

    val proj: Frame[RX, Int, Double] = {
      val projectedM: Mat[Double] = Mat(
        u.cols.zip(sigma.toSeq).filter(_._2 > 1E-4).map(x => x._1 * x._2): _*)
      Frame(projectedM, cov.rowIx, index.IndexIntRange(sigmaPos.length))
    }
    PCA(sigmaPos, proj, None)
  }

}
