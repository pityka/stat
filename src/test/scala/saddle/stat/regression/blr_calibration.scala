package stat.regression

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._

class BLRSuite extends FunSuite {
  /** Test of internal consistency
   * Analogous to testing whether the p-values are uniformly distributed under the null
   * 
   * Algorithm 1. in https://arxiv.org/pdf/1804.06788.pdf
   * 
   * The posterior of the parameter is distributed as the prior if the inference is performed on data
   * generated from the exact data model used for inference
   */
  test("calibration") {

    val rnd = new jdistlib.rng.RandomWELL44497b
    val X = {
      val N = 30
      Mat(vec.ones(N), jdistlib.Normal.random(N,10,10,rnd).toVec)
    }

    val X2 = {
      val N = 1
      Mat(vec.ones(N), jdistlib.Normal.random(N,10,10,rnd).toVec)
    }
    val lambda = 1.0
    val V0 = mat.diag(vec.ones(2)*1/lambda)
    def draw() = {
      // prior on sigma^2: [0,Infinity), practically bounded at 10
      val priorSigmaSquared = jdistlib.Uniform.random(0d,10d,rnd)

      // prior on the weights: MVN(0,sigma^2 * 1/lambda * I)
      val priorW1 = jdistlib.Normal.random(0d,priorSigmaSquared * V0.raw(0,0), rnd)
      val priorW2 = jdistlib.Normal.random(0d,priorSigmaSquared * V0.raw(1,1),rnd)
      val priorW = Vec(priorW1,priorW2)
      
      // data generating model: MVN(X*w, sigma^2 * I)
      val priorMeans = X mv priorW 
      val generatedY = priorMeans.map{ m =>
        jdistlib.Normal.random(m, priorSigmaSquared, rnd)
      }

      // inference returns posterior distribution of the weights
      // This is a Student's T distribution with DoF, mean, shape^2 parameters
      val (dof,posteriorMean, posteriorShape,_, prediction) = BayesianLinearRegression.bayesianLinearRegression(
        X,
        generatedY,
        lambda,
        vec.ones(2)
      )

      // take K samples from the above posterior distribution
      val K = 1000
      val posteriorW1 = (jdistlib.T.random(K,dof, rnd).toVec:Vec[Double]) * math.sqrt(posteriorShape.raw(0,0)) + posteriorMean.raw(0)
      val posteriorW2 = (jdistlib.T.random(K,dof, rnd).toVec:Vec[Double]) * math.sqrt(posteriorShape.raw(1,1)) + posteriorMean.raw(1)

      // rank statistic to compute inflation
      // these are to be uniformly distributed on [0,1)
      val w1St = posteriorW1.countif(_ < priorW1)
      val w2St = posteriorW2.countif(_ < priorW2)

      // same for the posterior predictive 
      val priorYSample  = jdistlib.Normal.random((X2 mv priorW).raw(0), priorSigmaSquared, rnd)
      val predictive = prediction(X2)
      val sampleFromPosteriorPredictive = {
        val mean = predictive.mean
        val shape = predictive.shape
        (jdistlib.T.random(K,dof, rnd).toVec:Vec[Double]) * math.sqrt(shape.raw(0,0)) + mean.raw(0)
      }
      val ppdSt = sampleFromPosteriorPredictive.countif(_ < priorYSample)
      

      (w1St/K.toDouble, w2St/K.toDouble, ppdSt / K.toDouble)
    }

    // perform M simulations and compute the inflation factor
    val M = 10000
    val uniforms =  (1 to M).map{ _ =>
      draw()
    }
    assert(stat.vis.QQUniform.computeQQError(uniforms.map(_._1)) < 1E-4)
    assert(stat.vis.QQUniform.computeQQError(uniforms.map(_._2)) < 1E-4)
    assert(stat.vis.QQUniform.computeQQError(uniforms.map(_._3)) < 1E-4)
  }
}