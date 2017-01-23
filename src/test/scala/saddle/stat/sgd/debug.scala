package stat.sgd

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._
import org.saddle.linalg._

class DebugSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.TRACE

  describe("debug") {

    it("L2") {

      val samples = 1000
      val nCol = 2
      val columns = nCol
      val design = mat.randn(samples, columns) mm mat.diag(Vec(1d, 2d))
      val betas: Vec[Double] = Vec(1.5, 1.5)
      val rng = new scala.util.Random(42)
      val ly = (design mv betas) + vec.randn(samples) * 2 //LinearRegression.generate(betas, design, () => rng.nextDouble)

      val lambda = 100
      val batch = 10

      val closed = stat.regression.LinearRegression.linearRegression(
        design,
        ly,
        lambda * samples / batch.toDouble,
        vec.ones(columns)
      )

      val validationBatch = {
        val design = mat.randn(samples, columns) mm mat.diag(Vec(1d, 2d))
        val ly = (design mv betas) + vec.randn(samples) * 2
        Batch(design, ly, vec.ones(columns), true)
      }

      val fitFista = stat.sgd.Sgd
        .optimize(
          MatrixData(design, ly, penalizationMask = vec.ones(columns)),
          sgd.LinearRegression,
          L2(lambda),
          FistaUpdater,
          kernel = IdentityFeatureMap,
          maxIterations = 400,
          minEpochs = 2d,
          convergedAverage = 200,
          epsilon = 1E-3,
          batchSize = batch,
          rng = scala.util.Random,
          normalize = true,
          validationBatch = Some(validationBatch)
        )
        .get
        .result

      val fitNewton = stat.sgd.Sgd
        .optimize(
          MatrixData(design, ly, penalizationMask = vec.ones(columns)),
          sgd.LinearRegression,
          L2(lambda),
          NewtonUpdater,
          kernel = IdentityFeatureMap,
          maxIterations = 400,
          minEpochs = 2d,
          convergedAverage = 200,
          epsilon = 1E-3,
          batchSize = batch,
          rng = scala.util.Random,
          normalize = true,
          validationBatch = Some(validationBatch)
        )
        .get
        .result

      val fitCD = stat.sgd.Sgd
        .optimize(
          MatrixData(design, ly, penalizationMask = vec.ones(columns)),
          sgd.LinearRegression,
          L2(lambda),
          CoordinateDescentUpdater,
          kernel = IdentityFeatureMap,
          maxIterations = 400,
          minEpochs = 2d,
          convergedAverage = 200,
          epsilon = 1E-3,
          batchSize = batch,
          rng = scala.util.Random,
          normalize = true,
          validationBatch = Some(validationBatch)
        )
        .get
        .result
      println(fitFista.estimatesV * fitFista.normalizer)
      println(fitNewton.estimatesV * fitNewton.normalizer)
      println(fitCD.estimatesV * fitCD.normalizer)
      println(closed)

    }

    ignore("L1") {

      val samples = 1000
      val nCol = 2
      val columns = nCol
      val design = mat.randn(samples, columns) mm mat.diag(Vec(1d, 2d))
      val betas: Vec[Double] = Vec(1.5, 1.5)
      val rng = new scala.util.Random(42)
      val ly = (design mv betas) + vec.randn(samples) * 2

      val lambda = 1000
      val batch = 10

      val validationBatch = {
        val design = mat.randn(samples, columns) mm mat.diag(Vec(1d, 2d))
        val ly = (design mv betas) + vec.randn(samples) * 2
        Batch(design, ly, vec.ones(columns), true)
      }

      val fitFista = stat.sgd.Sgd
        .optimize(
          MatrixData(design, ly, penalizationMask = vec.ones(columns)),
          sgd.LinearRegression,
          L1(lambda),
          FistaUpdater,
          kernel = IdentityFeatureMap,
          maxIterations = 400,
          minEpochs = 2d,
          convergedAverage = 200,
          epsilon = 1E-3,
          batchSize = batch,
          rng = scala.util.Random,
          normalize = false,
          validationBatch = Some(validationBatch)
        )
        .get
        .result

      val fitCD = stat.sgd.Sgd
        .optimize(
          MatrixData(design, ly, penalizationMask = vec.ones(columns)),
          sgd.LinearRegression,
          L1(lambda),
          CoordinateDescentUpdater,
          kernel = IdentityFeatureMap,
          maxIterations = 400,
          minEpochs = 2d,
          convergedAverage = 200,
          epsilon = 1E-3,
          batchSize = batch,
          rng = scala.util.Random,
          normalize = false,
          validationBatch = Some(validationBatch)
        )
        .get
        .result
      println(fitFista.estimatesV * fitFista.normalizer)
      println(fitCD.estimatesV * fitCD.normalizer)

    }
  }
}
