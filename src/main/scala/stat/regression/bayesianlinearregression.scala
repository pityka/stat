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
    def mean : Vec[Double]
    def variances: Vec[Double]
    def inverseCDF(p:Double) : Vec[Double]
    def covariances : Mat[Double]
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
    val aN = X.numRows.toDouble/2
    val bN = ((Y.innerM - Mat(wN).tmm(VNInverse).mm(Mat(wN))) * 0.5).raw(0,0)
    
    val dof = 2*aN 
    val estimatorCovariances = {
      VN * (dof/(dof-2d) * bN/aN) 
    }
    def estimatorMeans = wN 
    val estimatorVariances = estimatorCovariances.diag 
    def estimatorInverseCDF(p: Double) = 
      wN.zipMap(estimatorVariances){ case (w,va) =>
        val standardQuantile = jdistlib.T.quantile(p,dof,true,false)
        standardQuantile*math.sqrt(va)+w
      }
      
    def prediction(M:Mat[Double]) = {
      val m = M.mv(wN)
      val unit = mat.diag(vec.ones(M.numRows))
      val v = (unit + M.mm(VN.mmt(M))) * bN/aN 
      new PosteriorPredictive {
        val mean = m
        val covariances = v 
        val variances = covariances.diag.col(0)
        def inverseCDF(p: Double) = mean.zipMap(variances){ case (m1,v1) =>
          val standardQuantile = jdistlib.T.quantile(p,dof,true,false)
          standardQuantile*math.sqrt(v1) + m1
        }
      }
    }

  }
}