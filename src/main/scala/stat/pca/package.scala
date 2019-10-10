package stat

import org.saddle._
import org.saddle.linalg._

import org.nspl._
import org.nspl.saddle._
import org.nspl.data._

package object pca {
/* Does not standardize! */ 
  def fromData[RX: ORD: ST, CX: ORD: ST](
      data: Frame[RX, CX, Double],
      max: Int
  ): PCA[RX, CX] = {
    val demeaned = data.mapVec(_.demeaned)
    val standardized = demeaned.mapVec(v => v / math.sqrt(v.sampleVariance)).toMat
    val nonZero = standardized.singularValues(max).countif(_ > 1E-4)
    val SVDResult(u, sigma, vt) = demeaned.toMat.svd(nonZero)

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
    val covM = cov.reindexCol(cov.rowIx).toMat
    val nonZero = covM.eigenValuesSymm(max).countif(_ > 1E-4)
    val EigenDecompositionSymmetric(u, sigma2) = covM.eigSymm(nonZero)
    val sigma = sigma2.map(math.sqrt)
    val sigmaPos = sigma.filter(_ > 1E-4)

    val proj: Frame[RX, Int, Double] = {
      val projectedM: Mat[Double] = Mat(
        u.cols.zip(sigma.toSeq).filter(_._2 > 1E-4).map(x => x._1 * x._2): _*)
      Frame(projectedM, cov.rowIx, index.IndexIntRange(sigmaPos.length))
    }
    PCA(sigmaPos, proj, None)
  }

  def plot[RX: ORD: ST, CX: ORD: ST](
      pca: PCA[RX, CX],
      max: Int,
      groups: Either[Series[RX, String], Series[RX, Double]]) = {
    val PCA(eigen, project, loading) = pca
    val screeplot = xyplot(eigen)(xlab = "Component", ylab = "Eigenvalue")

    val colorValues = groups match {
      case scala.util.Left(x) =>
        Some(x.toVec.toSeq.distinct.zipWithIndex.toMap)
      case scala.util.Right(_) => None
    }

    val colornum: Series[RX, Double] =
      groups match {
        case scala.util.Left(groups) =>
          Series(project.rowIx.toSeq.map { rx =>
            rx -> (groups
              .first(rx)
              .flatMap(rx => colorValues.get(rx).toScalar)
              .map(_ + 1d)
              .getOrElse(0d))
          }: _*)
        case scala.util.Right(x) => x
      }

    val color = colorValues
      .map(cv => DiscreteColors(cv.size + 1))
      .getOrElse(
        HeatMapColors(colornum.values.min.get, colornum.values.max.get))

    val projections = 0 until max combinations (2) map { g =>
      val c1 = g(0)
      val c2 = g(1)
      xyplot(
        Frame(
          (project.col(c1, c2).toColSeq :+ ("color", colornum)).zipWithIndex
            .map(x => (x._2 -> x._1._2)): _*) -> point(labelText = true,
                                                       color = color))(
        xlab = s"PCA $c1",
        ylab = s"PCA $c2",
        extraLegend = colorValues.toList.flatten.map(
          x =>
            x._1 -> PointLegend(shape = Shape.rectangle(0, 0, 1, 1),
                                color = color(x._2 + 1d))))
    }

    val loadings = loading.toSeq.flatMap { loading =>
      0 until max map { c =>
        xyplot(HistogramData(loading.firstCol(c).toVec.toSeq, 20) -> bar())(
          xlab = s"Loadings of $c")
      }
    }

    val topLoadings = loading.toSeq.flatMap { loading =>
      0 until max map { c =>
        val l =
          if (loading.numRows >= 10)
            loading
              .firstCol(c)
              .sorted
              .take(
                ((0 until 5) ++ (loading.numRows - 5 until loading.numRows)).toArray
              )
          else loading.firstCol(c)

        barplotVertical(l.mapIndex(_.toString), main = s"Top loadings of $c")

      }
    }

    sequence(screeplot +: (projections ++ loadings ++ topLoadings).toList,
             TableLayout(4))

  }

}
