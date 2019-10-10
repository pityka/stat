package stat.sgd

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.csv._

class MultinomialLogisticSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  describe("small multinomial, R") {
    val frame = CsvParser
    .parse(scala.io.Source.fromFile(getClass.getResource("/").getPath + "/example.csv").getLines)
      .withColIndex(0)
      .withRowIndex(0)
      .mapValues(_.toDouble)

    val frame2 = Frame(
      frame.toColSeq.dropRight(1) :+ (("threeway",
                                       frame
                                         .firstCol("V2")
                                         .mapValues(x =>
                                           if (x > 0 && x < 1) 1d
                                           else if (x > 1) 2d
                                           else 0d))): _*)

    val frame3 = frame2.col("V3", "V4", "threeway")

    it("test") {
      // in R
      // library(nnet)
      // fit <- multinom(threeway ~ V3 + V4, data = d2)
      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame3,
                    "threeway",
                    stat.sgd.MultinomialLogisticRegression(3),
                    stat.sgd.L2(0d),
                    stat.sgd.NewtonUpdater,
                    normalize = false)
          .get

      assert(
        sgdresult.estimatesV.toSeq == Vector(-0.34917752762083,
                                             -0.5749749117630679,
                                             -0.019776398665027693,
                                             -0.22419832067619133,
                                             -9.438973754628996E-5,
                                             -0.11614483978196864))

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame3,
                    "threeway",
                    stat.sgd.MultinomialLogisticRegression(3),
                    stat.sgd.L2(0d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -0.34865372193566885,
          -0.5743069579284142,
          -0.019794364979082952,
          -0.2240910425696873,
          -8.710210710466075E-5,
          -0.11608502705030992).roundTo(3))

    }

  }

  describe("multinomial logistic regression sgd, logistic case (2 classes)") {

    val frame = CsvParser
      .parse(scala.io.Source.fromFile(getClass.getResource("/").getPath + "/example.csv").getLines)
      .withColIndex(0)
      .withRowIndex(0)
      .mapValues(_.toDouble)

    val frame2 = Frame(
      frame.toColSeq.dropRight(1) :+ (("V22",
                                       frame
                                         .firstCol("V22")
                                         .mapValues(x =>
                                           if (x > 3) 1.0 else 0.0))): _*)

    it("lambda=0") {

      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L2(0d),
                    stat.sgd.NewtonUpdater,
                    normalize = false)
          .get

      assert(
        sgdresult.estimatesV.roundTo(3) == Vec(-13.456195234435322,
                                               5.545366077585118,
                                               -0.731088037251173,
                                               7.797408156057576,
                                               -0.025861072669013906,
                                               -2.764485472939923,
                                               6.317573048576022,
                                               1.1802933728848795,
                                               1.1542083792196989,
                                               0.25385955873250254,
                                               1.3402161987974173,
                                               2.9665609394111656,
                                               -0.006116922069417704,
                                               2.282302915552224,
                                               -3.423815164958668,
                                               -0.6367978386700061,
                                               1.563218699134814,
                                               -0.2777165266403398,
                                               3.5125248737280463,
                                               0.9484255659960339,
                                               -4.40041491046502).roundTo(3))

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L2(0d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -13.69106195769109,
          5.6555276197613455,
          -0.7543112367792331,
          7.928194417153012,
          -0.03143304568336373,
          -2.8154302962879485,
          6.440145161745654,
          1.192771810669563,
          1.1694085652030592,
          0.2764403034011235,
          1.3950392178942699,
          3.029516593880796,
          -0.007277191768664082,
          2.328663545008925,
          -3.47924165524167,
          -0.6616643276666687,
          1.5873718414016307,
          -0.2677855862701128,
          3.603507115886141,
          0.9608240136332966,
          -4.467925611390463).roundTo(3))

    }

    it("lambda=50 L2") {

      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L2(50d),
                    stat.sgd.NewtonUpdater,
                    normalize = false)
          .get

      assert(
        sgdresult.estimatesV.roundTo(3) == Vec(
          -1.3499178021930225,
          0.2499826509687129,
          0.03615190890850822,
          0.15422369039129777,
          0.0214837656710709,
          -0.13662904446492477,
          0.19283397563284888,
          0.06531941049557777,
          0.01682902795087247,
          -0.06422399931772955,
          -0.020953083094977266,
          0.030671836401678487,
          -0.0011363959613358628,
          -0.004382640161271681,
          -0.1786464268844689,
          -0.023022740834233644,
          0.08226804023869569,
          0.01117549585737458,
          2.9710858736612853E-4,
          0.07927520831623122,
          -0.17511343222110562).roundTo(3))

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L2(50d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.3485667557223828,
          0.2498274857669027,
          0.03613596213200277,
          0.15416271422245675,
          0.02143244900317967,
          -0.1365723002556386,
          0.1927904822847066,
          0.06529906433492456,
          0.01686292502764712,
          -0.06416868525976877,
          -0.020991298035269796,
          0.0307605449742312,
          -0.001117332741048533,
          -0.0043497895565466635,
          -0.17853651394020303,
          -0.022983375992346348,
          0.08222906829938983,
          0.01115813633385176,
          4.151439251985735E-4,
          0.07920745409499311,
          -0.17507055324155124).roundTo(3))

    }

    it("lambda=50 L1") {

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L1(50d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(-1.152895286123268,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.00).roundTo(3))

    }

    it("lambda=5 L1") {

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.MultinomialLogisticRegression(2),
                    stat.sgd.L1(5d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.84681474955958,
          0.7542887253705821,
          0.0,
          0.44853992493337946,
          0.0,
          -0.2758590628464792,
          0.38578578649732703,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.44361454320895843,
          0.0,
          0.0,
          0.0,
          0.0,
          0.030576896387142928,
          -0.4682853584069547).roundTo(3))

    }

  }
}
