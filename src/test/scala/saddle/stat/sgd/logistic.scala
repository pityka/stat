package stat.sgd

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._

class LogisticSuite extends FunSpec with Matchers {
  describe("lasso alone") {

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
                              stat.sgd.LogisticRegression,
                              stat.sgd.L2(0d),
                              stat.sgd.NewtonUpdater)

      assert(
        sgdresult.estimatesV.roundTo(5) == Vec(-13.456195494927975,
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
                                               -4.400415060828859).roundTo(5))

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L2(0d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(5) == Vec(
          -13.813192243260982,
          5.712627615770042,
          -0.7718681701999867,
          7.989582626316036,
          -0.04404037029568946,
          -2.8491663318897786,
          6.505497043128322,
          1.2036907839149278,
          1.1761860182255015,
          0.2925057562172423,
          1.4257741553640995,
          3.0613099232429732,
          -0.012101105250586159,
          2.3498636661784342,
          -3.5092438814761016,
          -0.6750912381755376,
          1.5966366322564307,
          -0.2591454344568539,
          3.6598032732212302,
          0.963015026464971,
          -4.510022344777664).roundTo(5))

    }

    it("lambda=50 L2") {

      val sgdresult =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L2(50d),
                              stat.sgd.NewtonUpdater)
      assert(
        sgdresult.estimatesV.roundTo(5) == Vec(
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
          -0.15844169804207395).roundTo(5))

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L2(50d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(5) == Vec(
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
          -0.1584441386306449).roundTo(5))

    }

    it("lambda=50 L1") {

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L1(50d),
                              stat.sgd.FistaUpdater)
      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(-1.152811195316048,
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
                              stat.sgd.LogisticRegression,
                              stat.sgd.L1(5d),
                              stat.sgd.FistaUpdater)

      assert(
        sgdresultfista.estimatesV.roundTo(5) == Vec(
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
          -0.3898001431502568).roundTo(5))

    }

  }
}
