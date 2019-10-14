package stat.regression
import org.saddle._
import org.saddle.linalg._

/**  Bayesian linear regression as presented in 'Machine Learning' by Kevin Murphy 2012 MIT Press
  * Chapter 7
  *
  * Posterior of weights equation 7.75
  * Posterior predictive equation 7.76
  * Pior choice: a0=0,b0=0,w0=0,V0=(lambda*Identity)-1
  *
  */
object BayesianLinearRegression {

  trait PosteriorPredictive {
    def mean: Vec[Double]
    def variances: Vec[Double]
    def inverseCDF(p: Double): Vec[Double]
    def covariances: Mat[Double]
  }

  /** Assuming observation noise is known
    * Not adding -N/2*log(2Pi)
    * y ~ N(Xw,sigma2 * I) ; w ~ N(0, lambda^-1 * I)
    * This calculates P(y|sigma2,lambda)
    */
  def logMarginalLikelihood(
      lambda: Double,
      sigma2: Double,
      X: Mat[Double],
      y: Vec[Double],
      penalizationWeights: Vec[Double]
  ) = {
    val lDiag = (penalizationWeights) * lambda
    val K = ((X.mm(mat.diag(lDiag.map(d => 1 / d)))).mmt(X)) + mat.diag(
      vec.ones(X.numRows) * sigma2
    )
    val lnKdet = K.determinantPD.get / math.log10(math.E)
    val inv = K.invertPD.get
    lnKdet * (-0.5) + (-0.5) * y.vv(inv.mv(y))
  }

  def bayesianLinearRegression(
      X: Mat[Double],
      y: Vec[Double],
      lambda: Double,
      penalizationWeights: Vec[Double]
  ) = {

    val XtX = X.innerM
    val Y = Mat(y)
    val VNInverse = (XtX + (mat.diag(penalizationWeights) * lambda))
    val VN = VNInverse.invert
    val wN = VN.mm(X.tmm(Y)).col(0)
    val aN = X.numRows.toDouble / 2
    val bN = ((Y.innerM - Mat(wN).tmm(VNInverse).mm(Mat(wN))) * 0.5).raw(0, 0)

    val dof = 2 * aN
    val estimatorMeans = wN
    val estimatorShape = VN * bN / aN
    val estimatorVariances = (estimatorShape * (dof / (dof - 2d))).diag
    def estimatorInverseCDF(p: Double) =
      wN.zipMap(estimatorShape.diag) {
        case (w, va) =>
          val standardQuantile = jdistlib.T.quantile(p, dof, true, false)
          standardQuantile * math.sqrt(va) + w
      }

    def prediction(M: Mat[Double]) = {
      val m = M.mv(wN)
      val unit = mat.diag(vec.ones(M.numRows))
      val v = (unit + M.mm(VN.mmt(M)))
      new PosteriorPredictive {
        val mean = m
        val covariances = v * bN / aN * (dof / (dof - 2))
        val variances = covariances.diag
        val shape = v * bN / aN
        def inverseCDF(p: Double) = mean.zipMap((v * bN / aN).diag) {
          case (m1, v1) =>
            val standardQuantile = jdistlib.T.quantile(p, dof, true, false)
            standardQuantile * math.sqrt(v1) + m1
        }
      }
    }

    (dof, estimatorMeans, estimatorShape, estimatorVariances, prediction _)
  }
}
