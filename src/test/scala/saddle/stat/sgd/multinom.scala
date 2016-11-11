package stat.sgd

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._

class MultinomialLogisticSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  describe("small multinomial, R") {
    val frame = CsvParser
      .parse(CsvFile(getClass.getResource("/").getPath + "/example.csv"))
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
        stat.sgd.Sgd.optimize(frame3,
                              "threeway",
                              stat.sgd.MultinomialLogisticRegression(3),
                              stat.sgd.L2(0d),
                              stat.sgd.NewtonUpdater)

      assert(
        sgdresult.estimatesV.roundTo(3) == Vec(
          -0.34917752762083,
          -0.5749749117630679,
          -0.019776398665027707,
          -0.22419832067619133,
          -9.438973754628176E-5,
          -0.11614483978196864).roundTo(3))

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame3,
                              "threeway",
                              stat.sgd.MultinomialLogisticRegression(3),
                              stat.sgd.L2(0d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -0.34913971582315306,
          -0.574943754575812,
          -0.01976221391286555,
          -0.22417951713404402,
          -1.5787654058765205E-4,
          -0.11622077292369085).roundTo(3))

    }

  }

  describe("multinomial logistic regression sgd, logistic case (2 classes)") {

    val frame = CsvParser
      .parse(CsvFile(getClass.getResource("/").getPath + "/example.csv"))
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
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L2(0d),
                              stat.sgd.NewtonUpdater)

      assert(
        sgdresult.estimatesV.roundTo(3) == Vec(-13.456195494927975,
                                               5.54536621449407,
                                               -0.7310881106659288,
                                               7.797408274673465,
                                               -0.02586114237634524,
                                               -2.764485592346235,
                                               6.317573178174667,
                                               1.1802934437067043,
                                               1.15420838183681,
                                               0.2538596143543805,
                                               1.340216259958331,
                                               2.966561011646488,
                                               -0.00611698475953151,
                                               2.282302935891697,
                                               -3.4238152421921075,
                                               -0.6367978876344843,
                                               1.5632187042510126,
                                               -0.2777165325992358,
                                               3.5125250794521206,
                                               0.9484255220382978,
                                               -4.400415060828859).roundTo(3))

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L2(0d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -13.965551817191976,
          5.783828810930236,
          -0.7916670983244545,
          8.066621934992632,
          -0.056491782220990994,
          -2.889032195572826,
          6.590020730569251,
          1.2138333570394784,
          1.1873651870890003,
          0.3114379098511703,
          1.46534265740417,
          3.1013697600486507,
          -0.012478687784932623,
          2.3754909745185455,
          -3.5465070111854953,
          -0.6885231058722754,
          1.609729257407417,
          -0.2473813651630387,
          3.72456738007188,
          0.9670351814569454,
          -4.558141354592802).roundTo(3))

    }

    it("lambda=50 L2") {

      val sgdresult =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L2(50d),
                              stat.sgd.NewtonUpdater)
      assert(
        sgdresult.estimatesV.roundTo(3) == Vec(
          -1.3561202984178287,
          0.2594410065773155,
          0.03605859323645384,
          0.15547168948444257,
          0.022210330923179814,
          -0.12911547653084493,
          0.20333443325650846,
          0.06412955026816544,
          0.018469919307403607,
          -0.06483174218203624,
          -0.01999467409234319,
          0.030724112638692503,
          -7.351276994429093E-4,
          -0.004636385927574311,
          -0.19137568510449277,
          -0.021487131795048776,
          0.07822379105543227,
          0.011137070547289846,
          1.945350339377649E-5,
          0.08558699047822457,
          -0.15844169804207395).roundTo(3))

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L2(50d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.3562102827287341,
          0.2594525790493179,
          0.0360598255007527,
          0.15547609786362485,
          0.02221432261559209,
          -0.12911908255272067,
          0.20333753236727026,
          0.0641307875867453,
          0.018467571261175517,
          -0.06483563467764557,
          -0.019991844055722706,
          0.03071775198098998,
          -7.365388813866432E-4,
          -0.004638577703681529,
          -0.19138436897057307,
          -0.02148985372614914,
          0.07822634290273234,
          0.011138263506677085,
          1.0960471231854195E-5,
          0.08559228471086837,
          -0.1584441386306449).roundTo(3))

    }

    it("lambda=50 L1") {

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L1(50d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(-1.1524182159869036,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    -0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    -0.0,
                                                    -0.0,
                                                    0.0,
                                                    -0.0,
                                                    -0.0,
                                                    -0.0,
                                                    -0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                    -0.0).roundTo(3))

    }

    it("lambda=5 L1") {

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.MultinomialLogisticRegression(2),
                              stat.sgd.L1(5d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.8742473938491546,
          0.7939496733771211,
          0.0,
          0.4526335652561677,
          0.0,
          -0.2299279309871057,
          0.4143751263623059,
          0.0,
          0.0,
          -0.0,
          -0.0,
          0.0,
          0.0,
          -0.0,
          -0.4827569989422659,
          -0.0,
          0.0,
          0.0,
          0.0,
          0.0642348966774298,
          -0.3898001431502568).roundTo(3))

    }

  }
}
