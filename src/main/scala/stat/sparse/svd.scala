package stat.sparse

import stat.matops._
import org.saddle._
import org.saddle.linalg._
import slogging.StrictLogging

object Svd {
  def apply[T: MatOps](t: T,
                       k: Int,
                       maxIter: Int = 100,
                       minIter: Int = 5,
                       epsilon: Double = 1E-3): SVDResult = {

    val xxtOp = new LinearMap[T] {
      def numCols(t: T) = t.numRows
      def mv(t: T, v: Vec[Double]): Vec[Double] =
        t.mv(t.tmv(v))
    }

    val EigenDecompositionSymmetric(u, lambda) =
      Eigen.eigenDecompositionSymmetric(t, k, maxIter, minIter, epsilon)(xxtOp)

    val sigma = lambda.map(math.sqrt)
    val sigmainv = sigma.map(x => 1d / x)

    val utm = t.mmLeft(u.T)
    val vt = Mat(utm.rows.zip(sigmainv.toSeq).map(x => x._1 * x._2): _*).T
    SVDResult(u, sigma, vt)

  }
}
