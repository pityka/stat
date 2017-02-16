package stat.sgd.nn

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._
import stat.sgd._

class NeuralNetworkSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  describe("extractWs") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(nn.shapeW(0, 10) == (1, 10))
      assert(nn.extractWs(vec.ones(10), 10) == Vector(mat.ones(1, 10)))
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      assert(nn.shapeW(0, 10) == (2, 10))
      assert(nn.shapeW(1, 10) == (1, 2))
      assert(
        nn.extractWs(vec.ones(22), 10) == Vector(mat.ones(2, 10),
                                                 mat.ones(1, 2)))
    }
  }
  describe("outputLast") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(nn.outputLast(Vector(mat.ones(1, 10)), vec.ones(10)) == Vec(10d))
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      val ws = nn.extractWs(vec.ones(22), 10)
      assert(nn.outputLast(ws, vec.ones(10)).length == 1)
    }
  }
  describe("Backpropagation") {
    it("1") {
      val nn = NeuralNetwork(shape = Vector(1), rng = scala.util.Random)
      assert(
        nn.jacobiRow(Vector(mat.ones(1, 10)), vec.ones(10), 1.0).length == 10)
    }
    it("2") {
      val nn = NeuralNetwork(shape = Vector(2, 1), rng = scala.util.Random)
      assert(nn
        .jacobiRow(Vector(mat.ones(2, 10), mat.ones(1, 2)), vec.ones(10), 1.0)
        .length == 22)
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

    it("no hidden layer") {

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

      val expected = Vec(0.10612706212905593,
                         0.42976187684156536,
                         0.04467305307132958,
                         0.21151404355473594,
                         -0.056588192015039035,
                         -0.2527387037484237,
                         0.24322500909477074,
                         0.05174498110221614,
                         0.10998674261599989,
                         -0.03719806458525444,
                         -0.016503579771785223,
                         0.040935864015212126,
                         -0.05026160308716017,
                         0.011664262318738829,
                         -0.3800157990390685,
                         -0.0012879750584939057,
                         0.0020763089997701194,
                         0.031807540107561254,
                         0.012311862638598783,
                         9.30000373521594E-4,
                         -0.25307721384329807)

      assert(sgdresult.estimatesV.roundTo(3) == expected.roundTo(3))
    }

    it("1 hidden layer") {

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
        0.19569592905327485,
        0.39899708465792044,
        0.049758931544969735,
        0.17091045538073157,
        -0.024883464906010344,
        -0.24725585902646322,
        0.223677892130972,
        0.07089124983372902,
        0.05803570788765182,
        -0.06458153910785314,
        -0.015769687729580085,
        0.03465353322639814,
        -0.032077921043517435,
        -0.018491109821297657,
        -0.3375479580679482,
        -0.006893500926123951,
        0.05971804356706763,
        0.019941764119491424,
        -0.03151158648685562,
        0.037088914598244725,
        -0.22345569032127385,
        0.19565588985236151,
        0.39891545023397645,
        0.04974875091490768,
        0.17087548726415128,
        -0.024878373772756066,
        -0.2472052707129329,
        0.22363212784702374,
        0.07087674555145868,
        0.05802383383702394,
        -0.0645683257863907,
        -0.015766461266467517,
        0.0346464431464686,
        -0.03207135793138314,
        -0.01848732655782584,
        -0.33747889607683457,
        -0.006892090522398361,
        0.05970582531217026,
        0.01993768405351505,
        -0.031505139246240904,
        0.03708132624160323,
        -0.22340997149962202,
        0.19560506861074142,
        0.3988118327119193,
        0.049735828772396705,
        0.17083110268450963,
        -0.02487191166294039,
        -0.2471410596186444,
        0.22357403983130295,
        0.0708583354529309,
        0.05800876225761676,
        -0.0645515542877761,
        -0.015762365958427373,
        0.034637443799404095,
        -0.03206302745774859,
        -0.01848252451026675,
        -0.33739123658171266,
        -0.006890300315129363,
        0.059690316838720524,
        0.019932505278362683,
        -0.03149695584683257,
        0.037071694438330936,
        -0.22335194118860813,
        0.19568131997854482,
        0.3989672987637782,
        0.04975521694572742,
        0.17089769658893167,
        -0.024881607307844656,
        -0.2472374009045234,
        0.2236611941492711,
        0.07088595766658384,
        0.05803137541125334,
        -0.0645767179726588,
        -0.015768510492281382,
        0.03465094627398225,
        -0.03207552636546582,
        -0.018489729424648316,
        -0.3375227594684217,
        -0.006892986313120777,
        0.05971358550110239,
        0.019940275428723567,
        -0.03150923408676286,
        0.03708614583995973,
        -0.22343900892736873,
        0.19554439316425243,
        0.39868812382141455,
        0.04972040103506774,
        0.17077811196446543,
        -0.024864196554339562,
        -0.2470643980359815,
        0.22350468858805994,
        0.07083635564857214,
        0.0579907683090103,
        -0.06453153080676134,
        -0.015757476572888486,
        0.03462669948494343,
        -0.03205308171083002,
        -0.018476791348887345,
        -0.337286579968927,
        -0.0068881629878643725,
        0.05967180128260797,
        0.019926322342160855,
        -0.03148718569173466,
        0.037060195034825424,
        -0.22328265883897586,
        0.19760502139456015,
        0.19756854020289136,
        0.19752223142958733,
        0.1975917108231725,
        0.19746693804439175)

      assert(sgdresult.estimatesV.roundTo(2) == expected.roundTo(2))
    }

  }

}
