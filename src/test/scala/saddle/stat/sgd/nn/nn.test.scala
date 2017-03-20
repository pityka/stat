package stat.sgd.nn

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._
import stat.sgd._

class NeuralNetworkSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG
  val rng = scala.util.Random

  describe("extractWs") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(nn.shapeW(0, 10) == (1, 11))
      assert(nn.extractWs(vec.ones(11), 10) == Vector(mat.ones(1, 11)))
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      assert(nn.shapeW(0, 10) == (2, 11))
      assert(nn.shapeW(1, 10) == (1, 3))
      assert(
        nn.extractWs(vec.ones(25), 10) == Vector(mat.ones(2, 11),
                                                 mat.ones(1, 3)))
    }
  }
  describe("outputLast") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(nn.outputLast(Vector(mat.ones(1, 11)), vec.ones(10)) == Vec(11d))
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      val ws = nn.extractWs(vec.ones(25), 10)
      assert(nn.outputLast(ws, vec.ones(10)).length == 1)
    }
  }
  describe("Backpropagation") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(
        nn.jacobiRow(Vector(mat.ones(1, 11)), vec.ones(10), 1.0).length == 11)
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      assert(nn
        .jacobiRow(Vector(mat.ones(2, 11), mat.ones(1, 3)), vec.ones(10), 1.0)
        .length == 25)
      // println(nn
      //   .jacobiRow(Vector(mat.ones(2, 10), mat.ones(1, 2)), vec.ones(10), 1.0)
      //   .toSeq)
    }
    // it("2 derivative") {
    //   val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
    //   val ws1 = Vector(mat.ones(2, 10), mat.ones(1, 2))
    //   val ws2 = Vector(mat.ones(2, 10), Mat(Vec(1.1d), Vec(1d)))
    //   val row = (0 until 10).toVec.map(_.toDouble)
    //   val jacobiAt1 = nn.jacobiRow(ws1, row, 1.0)
    //   println(jacobiAt1.toSeq)
    //   val funAt1 = math.pow(nn.outputLast(ws1, row).raw(0) - 1.0, 2d) * 0.5
    //   val funAt2 = math.pow(nn.outputLast(ws2, row).raw(0) - 1.0, 2d) * 0.5
    //   val move =
    //   // assert(jacobiAt1.countif(_ != 0.0) == 1.0)
    //   assert(funAt2 - funAt1 == jacobiAt1.raw(1))
    // }
  }

  describe("full") {
    val frame = CsvParser
      .parse(CsvFile(getClass.getResource("/").getPath + "/example.csv"))
      .withColIndex(0)
      .withRowIndex(0)
      .mapValues(_.toDouble)

    ignore("no hidden layer") {

      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame,
                    "V22",
                    NeuralNetwork(Vector(1)),
                    stat.sgd.L2(50d),
                    stat.sgd.SimpleUpdater(1E-3),
                    normalize = false,
                    maxIterations = 10000,
                    stop = RelativeStopTraining(1E-10))
          .get

      val expected = Vec(0.18956507555854826,
                         0.9908290275920091,
                         0.07126780268902774,
                         0.5012350204907906,
                         -0.03062930660071736,
                         -0.6078325516061882,
                         0.5129953519293573,
                         0.11335359892271299,
                         0.27825320160886635,
                         -0.06519987938150386,
                         0.019298778282373544,
                         0.1551559860751918,
                         -0.06201141807904416,
                         -0.012408580249266755,
                         -0.8368865405863605,
                         -0.040143187164688864,
                         0.01902239997407996,
                         0.018298914070841013,
                         0.04049657671650208,
                         0.022508571215452137,
                         -0.6871805367784808)

      assert(sgdresult.estimatesV.roundTo(3) == expected.roundTo(3))
    }

    ignore("1 hidden layer") {

      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame,
                    "V22",
                    NeuralNetwork(Vector(5, 1)),
                    stat.sgd.L2(50d),
                    stat.sgd.SimpleUpdater(1E-3),
                    normalize = false,
                    maxIterations = 10000,
                    stop = RelativeStopTraining(1E-10))
          .get

      val expected = Vec(
        0.002842635305663832,
        -0.031970874234398955,
        -0.0030518914993511725,
        -0.016323862524940697,
        0.0028876293106434724,
        0.01954883579783252,
        -0.0176618704354265,
        -0.0021643062350828364,
        -0.010541894281102646,
        0.001279046024311421,
        6.624563757580389E-4,
        -0.006555679505824469,
        0.0015109825667604078,
        -7.812178567606836E-4,
        0.028459336932421787,
        -2.5260532851726145E-4,
        0.001315443223195194,
        -0.0017060464684984296,
        -0.003915234059653147,
        8.145652290407765E-4,
        0.02108648878245415,
        0.002843130891422935,
        -0.03197408403669495,
        -0.0030522363512620492,
        -0.016325484873003303,
        0.0028879317230892892,
        0.019550797559362506,
        -0.017663701519343863,
        -0.0021645104643812176,
        -0.010543016733685809,
        0.001279141380705865,
        6.625783854894797E-4,
        -0.006556384364631411,
        0.0015111516966584487,
        -7.813294087664179E-4,
        0.028462244445707,
        -2.526452981831013E-4,
        0.0013156123486243027,
        -0.0017062678409991926,
        -0.003915698220201341,
        8.14693445118838E-4,
        0.021088599706264862,
        0.3756285128516501,
        -0.5110905137753068,
        -0.1088766292443535,
        -0.23455479838480806,
        0.06155596579991915,
        0.3055354188818978,
        -0.3705239968616189,
        -0.027788929023503744,
        -0.2615275618308308,
        -0.03943163550719674,
        0.09355082111027312,
        -0.17721878838832184,
        0.06900403934260799,
        -0.05931444145653167,
        0.51710704147844,
        -0.015587975289503769,
        0.0947142938232794,
        -0.10235332304942128,
        -0.1693626680686841,
        0.09689175869359103,
        0.3458661375495475,
        0.002872594652742695,
        -0.03216439724343855,
        -0.0030726982412491864,
        -0.016421669393331485,
        0.0029058661353808767,
        0.019667112352089563,
        -0.01777229170942107,
        -0.0021766147322775705,
        -0.010609593407299884,
        0.0012847824676926378,
        6.698340222592319E-4,
        -0.006598194870828019,
        0.0015211869148512175,
        -7.879560677902503E-4,
        0.028634653978606234,
        -2.5502072100179826E-4,
        0.0013256539476515254,
        -0.001719412886969861,
        -0.003943246409885287,
        8.223130214361995E-4,
        0.021213757270829917,
        0.49718856772623254,
        0.7629791065789688,
        -0.0037581015963969195,
        0.4458136884174978,
        0.0016762144948754536,
        -0.4777594559452614,
        0.33112664764735783,
        0.08860249451416352,
        0.1353759903120131,
        -0.11196269012275849,
        0.09933563765391121,
        0.05764620288185415,
        -0.02639100329322329,
        -0.0471143338255076,
        -0.5899597513768446,
        -0.04853468985715929,
        0.056227370485335736,
        -0.05603504722564097,
        -0.07504357677535783,
        0.07875231921249423,
        -0.5354100482651313,
        -0.07073630611325388,
        -0.07074225313536343,
        -1.0756707400391503,
        -0.07109488292183061,
        1.349893051877269)

      assert(sgdresult.estimatesV.roundTo(2) == expected.roundTo(2))
    }

  }

  describe("overfit") {
    ignore("linear") {

      val x1 = vec.randn(200)
      val x2 = vec.randn(200)
      val y = x1 * 3 + x2 * (-2) + 5

      val sgdresult =
        stat.sgd.Sgd
          .optimize(Frame("x1" -> x1, "x2" -> x2, "y" -> y),
                    "y",
                    NeuralNetwork(Vector(1)),
                    stat.sgd.NoPenalty,
                    stat.sgd.SimpleUpdater(1E-4),
                    normalize = false,
                    maxIterations = 10000,
                    stop = RelativeStopTraining(1E-10))
          .get

      println(sgdresult)

    }

    it("non-linear 1") {

      val x1 = vec.randn(200)
      val x2 = vec.randn(200)
      val xor = x1.zipMap(x2)((x, y) => if (x * y > 0) 0d else 1)
      val y = x1.map(x => math.sin(x * 5))
      // val y = xor //(x1 * 3 + x2 * (-2)).map(math.sin) // //(x1 * 3 + x2 * (-2)) * 0d + x1 * x2 * 10 //+ xor * 10
      val f = Frame("x1" -> x1, "x2" -> x2, "y" -> y)
      println(f)
      println(f.toMat)
      // val sgdresult = stat.sgd.Sgd
      //   .optimizeFrame(
      //     data = f,
      //     yKey = "y",
      //     obj = NeuralNetwork(Vector(100, 10, 1)),
      //     pen = stat.sgd.L2(0.1d),
      //     upd = stat.sgd.RMSPropUpdater(0.01),
      //     addIntercept = false,
      //     maxIterations = 50000,
      //     minEpochs = 100d,
      //     convergedAverage = 1000,
      //     stop = TrainingStall(5000, 1E-3),
      //     rng = rng,
      //     kernel = IdentityFeatureMap,
      //     normalize = false,
      //     batchSize = Some(10)
      //   )
      //   .get

      val (eval, sgdresult) =
        stat.sgd.Cv.fitWithCV(
          data = f,
          yKey = "y",
          addIntercept = false,
          obj = NeuralNetwork(Vector(100, 10, 1)),
          pen = stat.sgd.L2(1d),
          upd = stat.sgd.RMSPropUpdater(0.01),
          outerSplit = stat.crossvalidation.Split(0.6, rng),
          split = stat.crossvalidation.KFold(5, rng, 1),
          search =
            stat.crossvalidation.HyperParameterSearch.GridSearch(-6, -3, 5),
          kernelFactory = IdentityFeatureMapFactory, //RbfFeatureMapFactory(rbfCenters),
          bootstrapAggregate = None,
          maxIterations = 50000,
          minEpochs = 100.0,
          convergedAverage = 1000,
          batchSize = Some(10),
          stop = TrainingStall(5000, 1E-3),
          rng = rng,
          normalize = false,
          warmStart = true)
      println(eval)

      val predicted: Vec[Double] =
        sgdresult.predict(f.toMat.col(0, 1))
      println(Frame("pr" -> predicted, "tr" -> y))

      println(sgdresult)

    }

  }

}
