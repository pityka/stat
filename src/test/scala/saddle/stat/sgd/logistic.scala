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
        sgdresult.estimates.roundTo(5) == Vec(-13.456195494927975,
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
        sgdresultfista.estimates.roundTo(5) == Vec(
          -13.61186445590838,
          5.618215558980256,
          -0.748716873867643,
          7.881086022098782,
          -0.033460678481760514,
          -2.801386548543574,
          6.399026577845969,
          1.189874393996833,
          1.1635620709680088,
          0.270225292613257,
          1.3775940904039072,
          3.0073438555436827,
          -0.00874267691012021,
          2.311874861704444,
          -3.460902911271312,
          -0.653534070864105,
          1.5772347512345877,
          -0.26946038259081573,
          3.5756044760726997,
          0.9554850661236307,
          -4.447833450773019).roundTo(5))

    }

    it("lambda=50 L2") {

      val sgdresult =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L2(50d),
                              stat.sgd.NewtonUpdater)
      assert(
        sgdresult.estimates.roundTo(5) == Vec(-1.3561202984178287,
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
        sgdresultfista.estimates.roundTo(5) == Vec(
          -1.3562587322719628,
          0.25945959634514604,
          0.036060578908151085,
          0.15547880952119436,
          0.02221684095413398,
          -0.12912123937250566,
          0.2033393214477357,
          0.06413148562125626,
          0.018466200452706655,
          -0.0648380368244255,
          -0.01999005124116623,
          0.030713851227698808,
          -7.373849477994779E-4,
          -0.0046399586584294425,
          -0.1913895874265109,
          -0.02149156963947883,
          0.07822789393753939,
          0.01113896479810461,
          5.762855711526553E-6,
          0.08559549155381534,
          -0.15844559032246183).roundTo(5))

    }

    it("lambda=50 L1") {

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L1(50d),
                              stat.sgd.FistaUpdater)
      assert(
        sgdresultfista.estimates.roundTo(5) == Vec(-1.152811195316048,
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
                                                   -0.0).roundTo(5))

    }

    it("lambda=5 L1") {

      val sgdresultfista =
        stat.sgd.Sgd.optimize(frame2,
                              "V22",
                              stat.sgd.LogisticRegression,
                              stat.sgd.L1(5d),
                              stat.sgd.FistaUpdater)
      assert(
        sgdresultfista.estimates.roundTo(5) == Vec(
          -1.8743567848728366,
          0.794013937149188,
          0.0,
          0.45267799383212315,
          0.0,
          -0.22993320486740523,
          0.4143921691603501,
          0.0,
          0.0,
          -0.0,
          -0.0,
          0.0,
          0.0,
          -0.0,
          -0.48279351637889384,
          -0.0,
          0.0,
          0.0,
          0.0,
          0.06424258003645346,
          -0.3898071576919604).roundTo(5))

    }

  }
}
