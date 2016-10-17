package stat

import org.saddle._
import org.saddle.linalg._

import org.nspl._
import org.nspl.saddle._
import org.nspl.data._

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

  def plot[RX: ORD: ST, CX: ORD: ST](pca: PCA[RX, CX], max: Int) = {
    val PCA(eigen, project, loading) = pca
    val screeplot = xyplot(eigen)(xlab = "Component", ylab = "Eigenvalue")

    val projections = 0 until max combinations (2) map { g =>
      val c1 = g(0)
      val c2 = g(1)
      xyplot(project.col(c1, c2) -> point(labelText = true))(xlab = s"PCA $c1",
                                                             ylab = s"PCA $c2")
    }

    val loadings = loading.toSeq.flatMap { loading =>
      0 until max map { c =>
        xyplot(HistogramData(loading.firstCol(c).toVec.toSeq, 20) -> bar())(
          xlab = s"Loadings of $c")
      }
    }

    sequence(screeplot +: (projections ++ loadings).toList, TableLayout(4))

  }

}
