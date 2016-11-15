package stat.sgd

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat._

class LMRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG
  test("random ") {
    val samples = 1000
    val nCol = 5
    val columns = nCol * samples
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(
        nCol * samples - 20)
    val rng = new scala.util.Random(23)
    val ly = LinearRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    val ds = DataSource.fromMat(design,
                                ly,
                                (0 until samples).toVec,
                                samples,
                                vec.ones(columns),
                                42)

    val fitFista = Cv.fitWithCV(
      design,
      ly,
      sgd.LinearRegression,
      L1(1.0),
      FistaUpdater,
      0.9,
      stat.crossvalidation.KFold(5, 42, 1),
      stat.crossvalidation.RandomSearch(() => rng.nextDouble),
      hMin = -6d,
      hMax = 15d,
      hN = 10,
      penalizationMask = vec.ones(columns),
      maxIterations = 3000,
      minEpochs = 1,
      convergedAverage = 2,
      epsilon = 1E-2,
      42
    )

    assert(fitFista._1.misc > 0.95)

  }
}

class LRRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG
  test("random ") {
    val samples = 1000
    val columns = 10000
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(columns - 20)
    val rng = new scala.util.Random(23)
    val ly = LogisticRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    // println(betas)
    // println(ly)

    val ds = DataSource.fromMat(design,
                                ly,
                                (0 until samples).toVec,
                                samples,
                                vec.ones(columns),
                                42)

    val fitFista = Cv.fitWithCV(
      design,
      ly,
      sgd.LogisticRegression,
      ElasticNet(1.0, 1.0),
      FistaUpdater,
      0.8,
      stat.crossvalidation.KFold(5, 42, 1),
      stat.crossvalidation.RandomSearch2D(() => rng.nextDouble),
      hMin = -2d,
      hMax = 15d,
      hN = 20,
      penalizationMask = vec.ones(columns),
      maxIterations = 5000,
      minEpochs = 1,
      convergedAverage = 2,
      epsilon = 1E-3,
      42
    )
    assert(fitFista._1.misc._1 > 0.75)

  }
}

class MNLRRandomSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.TRACE
  test("random ") {
    val samples = 1000
    val columns = 5000
    val design = mat.randn(samples, columns)
    val betas: Vec[Double] = vec.randn(20) concat vec.zeros(columns - 20)
    val rng = new scala.util.Random(23)
    val ly = LogisticRegression
        .generate(betas, design, () => rng.nextDouble) + vec.randn(samples) * 0

    println(betas)
    println(ly)

    val ds = DataSource.fromMat(design,
                                ly,
                                (0 until samples).toVec,
                                samples,
                                vec.ones(columns),
                                42)

    val fitFista = Cv.fitWithCV(
      design,
      ly,
      sgd.MultinomialLogisticRegression(2),
      ElasticNet(1.0, 1.0),
      FistaUpdater,
      0.8,
      stat.crossvalidation.KFold(5, 42, 1),
      stat.crossvalidation.RandomSearch2D(() => rng.nextDouble),
      hMin = -2d,
      hMax = 15d,
      hN = 20,
      penalizationMask = vec.ones(columns),
      maxIterations = 5000,
      minEpochs = 1,
      convergedAverage = 2,
      epsilon = 1E-3,
      42
    )
    println(fitFista)
    assert(fitFista._1.misc > 0.75)

  }
}