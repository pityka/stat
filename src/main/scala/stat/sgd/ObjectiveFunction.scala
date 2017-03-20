package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

trait ObjectiveFunction[E, @specialized(Double) P] {

  def jacobi1D[T: MatOps](b: Vec[Double],
                          batch: Batch[T],
                          i: Int,
                          xmbv: Vec[Double]): Double

  def hessian1D[T: MatOps](p: Vec[Double],
                           batch: Batch[T],
                           i: Int,
                           old: Option[Double],
                           xmvb: Vec[Double]): Double
  def jacobi[T: MatOps](b: Vec[Double], batch: Batch[T]): Vec[Double]
  def hessian[T: MatOps](p: Vec[Double], batch: Batch[T]): Mat[Double]
  def minusHessianLargestEigenValue[T: MatOps](p: Vec[Double],
                                               batch: Batch[T]): Double
  def apply[T: MatOps](b: Vec[Double],
                       batch: Batch[T],
                       work: Option[Array[Double]] = None): Double
  def start(cols: Int): Vec[Double]

  def predictMat(estimates: Vec[Double], data: Mat[Double]): Vec[P]
  def predict[T: MatOps](estimates: Vec[Double], data: T): Vec[P]

  def generate(estimates: Vec[Double],
               data: Mat[Double],
               rng: () => Double): Vec[Double]

  def eval[T: MatOps](est: Vec[Double], batch: Batch[T]): E

  def adaptPenalizationMask[T](batch: Batch[T]): Vec[Double]
  def adaptParameterNames(s: Seq[String]): Seq[String]

}
