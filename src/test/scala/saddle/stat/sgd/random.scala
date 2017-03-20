package stat.sgd

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._
import stat.crossvalidation.Split

class LMOverfitRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.TRACE
  test("random ") {
    val samples = 100
    val nCol = 20
    val columns = nCol * samples
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(
        nCol * samples - 20)
    val rng = new scala.util.Random(42)
    val ly = vec.randn(samples)

    val fitFista = Cv.fitWithCV(
      MatrixData(design, ly, penalizationMask = vec.ones(columns)),
      sgd.LinearRegression,
      L1(1.0),
      FistaUpdater,
      Split(0.9, rng),
      stat.crossvalidation.KFold(5, rng, 1),
      stat.crossvalidation.HyperParameterSearch.GridSearch1D(-6d, 15d, 3),
      IdentityFeatureMapFactory,
      bootstrapAggregate = None,
      maxIterations = 3000,
      minEpochs = 1,
      convergedAverage = 2,
      stop = RelativeStopTraining(1E-6),
      batchSize = design.numRows,
      maxEvalSize = design.numRows,
      rng = rng,
      normalize = true,
      warmStart = true
    )
    println(fitFista)

    assert(fitFista._1.misc < 0.1)

  }
}

class LMRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.TRACE
  test("random ") {
    val samples = 1000
    val nCol = 5
    val columns = nCol * samples
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(
        nCol * samples - 20)
    val rng = new scala.util.Random(42)
    val ly = LinearRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    val fitFista = Cv.fitWithCV(
      MatrixData(design, ly, penalizationMask = vec.ones(columns)),
      sgd.LinearRegression,
      L1(1.0),
      CoordinateDescentUpdater(),
      Split(0.9, rng),
      stat.crossvalidation.KFold(5, rng, 1),
      stat.crossvalidation.HyperParameterSearch.GridSearch1D(-1d, 15d, 3),
      IdentityFeatureMapFactory,
      bootstrapAggregate = None,
      maxIterations = 3000,
      minEpochs = 1,
      convergedAverage = 2,
      stop = RelativeStopTraining(1E-4),
      batchSize = design.numRows,
      maxEvalSize = design.numRows,
      rng = rng,
      normalize = true,
      warmStart = true
    )
    println(fitFista)

    assert(fitFista._1.misc > 0.95)

  }
}

class LRRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.TRACE
  test("random ") {
    val samples = 1000
    val columns = 10000
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(columns - 20)
    val rng = new scala.util.Random(42)
    val ly = LogisticRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    val fitFista = Cv.fitWithCV(
      MatrixData(design, ly, penalizationMask = vec.ones(columns)),
      sgd.LogisticRegression,
      ElasticNet(1.0, 1.0),
      CoordinateDescentUpdater(true),
      Split(0.8, rng),
      stat.crossvalidation.KFold(5, rng, 1),
      stat.crossvalidation.HyperParameterSearch
        .RandomSearch2D(-1d, 5d, -1d, 5d, 5)(() => rng.nextDouble),
      IdentityFeatureMapFactory,
      bootstrapAggregate = None,
      maxIterations = 5000,
      minEpochs = 1,
      convergedAverage = 2,
      stop = RelativeStopTraining(1E-4),
      batchSize = design.numRows,
      maxEvalSize = design.numRows,
      rng,
      normalize = true,
      warmStart = true
    )
    println(fitFista)

    assert(fitFista._1.misc._1 > 0.75)

  }
}

class MNLRRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG
  test("random ") {
    val samples = 1000
    val columns = 5000
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(columns - 20)
    val rng = new scala.util.Random(42)
    val ly = LogisticRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    val fitFista = Cv.fitWithCV(
      MatrixData(design, ly, penalizationMask = vec.ones(columns)),
      sgd.MultinomialLogisticRegression(2),
      ElasticNet(1.0, 1.0),
      FistaUpdater,
      Split(0.8, rng),
      stat.crossvalidation.KFold(5, rng, 1),
      stat.crossvalidation.HyperParameterSearch
        .RandomSearch2D(-1d, 5d, -1d, 5d, 4)(() => rng.nextDouble),
      IdentityFeatureMapFactory,
      bootstrapAggregate = None,
      maxIterations = 5000,
      minEpochs = 1,
      convergedAverage = 2,
      stop = RelativeStopTraining(1E-4),
      batchSize = design.numRows,
      maxEvalSize = design.numRows,
      rng,
      normalize = true,
      warmStart = true
    )
    println(fitFista)
    assert(fitFista._1.misc.accuracy > 0.7)

  }
}
