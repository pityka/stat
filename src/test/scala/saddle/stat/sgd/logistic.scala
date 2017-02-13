package stat.sgd

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._

class LogisticSuite extends FunSpec with Matchers {
  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  describe("logistic regression") {

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
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(0d),
                    stat.sgd.NewtonUpdater,
                    normalize = false)
          .get

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
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(0d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -13.688804926875768,
          5.654477954872492,
          -0.7545545044725792,
          7.925678893174654,
          -0.032317731327964366,
          -2.8157325748912383,
          6.439580989966532,
          1.192519566885812,
          1.1693339022926035,
          0.2765158441714768,
          1.3955655797301085,
          3.028469144835891,
          -0.007057456826890699,
          2.327658194811086,
          -3.4787277089521216,
          -0.6612343398929794,
          1.5866413698729973,
          -0.2668152054042013,
          3.60280280651425,
          0.9606160508617946,
          -4.467251342727571).roundTo(3))

    }

    it("lambda=50 L2") {

      val sgdresult =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(50d),
                    stat.sgd.NewtonUpdater,
                    normalize = false)
          .get

      assert(
        sgdresult.estimatesV.roundTo(2).toSeq == Vec(
          -1.3499178021930225,
          0.2499826509687129,
          0.03615190890850822,
          0.15422369039129769,
          0.021483765671070906,
          -0.13662904446492477,
          0.19283397563284888,
          0.06531941049557778,
          0.01682902795087246,
          -0.06422399931772955,
          -0.020953083094977266,
          0.030671836401678487,
          -0.0011363959613358645,
          -0.004382640161271688,
          -0.1786464268844689,
          -0.023022740834233644,
          0.0822680402386956,
          0.011175495857374578,
          2.9710858736613503E-4,
          0.07927520831623122,
          -0.17511343222110562).roundTo(2).toSeq)

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(50d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.350120412630522,
          0.2500081740307033,
          0.03615474326725601,
          0.1542338546006677,
          0.02149299614964105,
          -0.13663831518572878,
          0.19284032892757225,
          0.06532251111057269,
          0.016823583011355236,
          -0.06423328397747613,
          -0.020946335645244105,
          0.03065680154511915,
          -0.0011397289792719177,
          -0.004388264697492141,
          -0.17866449108266344,
          -0.02302985072071194,
          0.08227459036692732,
          0.011178196081010498,
          2.7725402458445453E-4,
          0.07928631861297186,
          -0.1751205684554714).roundTo(3))

      val sgdresultcd =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(50d),
                    stat.sgd.CoordinateDescentUpdater(),
                    normalize = false)
          .get

      assert(
        sgdresultcd.estimatesV.roundTo(3) == Vec(
          -1.3499173062100476,
          0.24998259809281326,
          0.03615188593796751,
          0.1542236895756601,
          0.021483723415875473,
          -0.1366290400638897,
          0.19283399065308215,
          0.0653194254831266,
          0.016829023689372013,
          -0.06422398545496838,
          -0.020953096721210342,
          0.030671844508179585,
          -0.0011364043372516005,
          -0.00438263293539544,
          -0.17864639261983187,
          -0.023022735575774958,
          0.08226802942997603,
          0.011175489221420523,
          2.9714835811763527E-4,
          0.07927519279819666,
          -0.17511341901529537).roundTo(3))

      val sgdresultsimple =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L2(50d),
                    stat.sgd.SimpleUpdater(1E-4),
                    normalize = false)
          .get

      assert(
        sgdresultsimple.estimatesV.roundTo(3) == Vec(
          -1.3069598115262047,
          0.24538362823204107,
          0.03566838115069995,
          0.1523845845240208,
          0.019881350102697412,
          -0.1349438767579,
          0.19158873994262846,
          0.06471020693683216,
          0.017875085309497225,
          -0.06257512077919207,
          -0.02212347728443545,
          0.033340238300117064,
          -5.483573138363978E-4,
          -0.003387567864031351,
          -0.1753654113830324,
          -0.021787709907223993,
          0.08107340849020327,
          0.010679970771991575,
          0.0038572290827267996,
          0.07726710994978811,
          -0.1737841581915465).roundTo(3))

    }

    it("lambda=50 L1") {

      val sgdresultfista =
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L1(50d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get
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
        stat.sgd.Sgd
          .optimize(frame2,
                    "V22",
                    stat.sgd.LogisticRegression,
                    stat.sgd.L1(5d),
                    stat.sgd.FistaUpdater,
                    normalize = false)
          .get

      assert(
        sgdresultfista.estimatesV.roundTo(3) == Vec(
          -1.8467933770460179,
          0.7542768266024502,
          0.0,
          0.4485308714903972,
          0.0,
          -0.2758570642925921,
          0.3857827436291803,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.44360788090632775,
          0.0,
          0.0,
          0.0,
          0.0,
          0.03057584815049286,
          -0.4682820839884609).roundTo(3))

    }

  }
}
