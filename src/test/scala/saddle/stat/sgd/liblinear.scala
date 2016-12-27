package stat.sgd

import org.saddle._
import org.saddle.linalg._
import org.scalatest.FunSuite
import stat.sparse._
import stat.crossvalidation.Split

class LibLinearSuite extends FunSuite {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  def readLibLinearToDense(s: scala.io.Source) = {
    val mat = Frame(s.getLines.zipWithIndex.map {
      case (line, idx) =>
        val spl = line.split(" ")
        val y = spl(0)
        val x = spl.drop(1).map { x =>
          val spl = x.split(":")
          println(x)
          val i = spl(0).toInt
          val v = spl(1).toDouble
          (i, v)
        }
        idx -> Series(((0 -> y.toDouble) +: x): _*)
    }.toSeq: _*).T.toMat

    val y = mat.col(0)
    val x = mat.takeCols(1 until mat.numCols: _*)
    (y, x)
  }

  def readLibLinearToSparse(s: scala.io.Source): (Vec[Double], SMat) = {
    val (y, x) = s.getLines.map { line =>
      val spl = line.split(" ")
      val y = spl(0)
      val x = spl.drop(1).map { x =>
        val spl = x.split(":")
        val i = spl(0).toInt
        val v = spl(1).toDouble
        (i, v)
      }
      y.toDouble -> Series((0 -> 1d) +: x: _*)
    }.toSeq.unzip

    val maxCol: Int = x.map(_.index.toVec.max).max.get

    println(maxCol)

    (y.toVec, x.toVector.map(x => SVec(x, maxCol + 1)))
  }

  test("file ") {
    val file = System.getProperty("user.home") + "/Downloads/aloi"
    val (y, x) = readLibLinearToSparse(scala.io.Source.fromFile(file))
    slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
    slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

    println(y.toSeq.groupBy(x => x).toSeq.map(x => x._1 -> x._2.size))
    println(y.toSeq.distinct.size)
    println(numCols(x))
    println(numRows(x))

    val idx = y.find(_ <= 50d)
    val ymulti = y(idx) //.map(_ + 1d)
    // val ybin = y.map(x => if (x > 0) 1d else 0d)
    // println(ybin)
    println(ymulti)

    val sf =
      SparseMatrixData(idx.map(i => x(i)).toSeq.toVector,
                       ymulti,
                       penalizationMask = vec.ones(numCols(x)))

    val rng = new scala.util.Random(42)
    val fitFista = Cv.fitWithCV(
      sf,
      stat.sgd.MultinomialLogisticRegression(51),
      stat.sgd.ElasticNet(1d, 1d),
      stat.sgd.FistaUpdater,
      Split(0.6, rng),
      stat.crossvalidation.KFoldStratified(5,
                                           rng,
                                           1,
                                           (0 to 50).toSeq.distinct
                                             .map(_.toDouble)
                                             .map(y => ymulti.find(_ == y))),
      stat.crossvalidation.RandomSearch2D(() => rng.nextDouble),
      hMin = -4d,
      hMax = 2d,
      hN = 20,
      maxIterations = 50000,
      minEpochs = 1.0,
      convergedAverage = 10,
      epsilon = 1E-3,
      batchSize = 16,
      maxEvalSize = 2000,
      rng
    )

    println(fitFista)

  }
}
