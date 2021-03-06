package stat.regression

import org.saddle._
import org.scalatest.FunSuite
import stat._

class LogRegSuite extends FunSuite {
  test("1") {
    val y: Vec[Double] =
      array.linspace(-1d, 1d).map(v => if (v > 0.0) 1.0 else 0.0).toVec
    val x: Vec[Double] = Vec(-0.042225672555113176,
                             -0.4454131082238157,
                             -0.8621926316533379,
                             -0.5390278113253716,
                             0.06727151600571332,
                             -0.6289341775366835,
                             -0.46930149977970304,
                             0.08308701640610404,
                             -0.852912858895757,
                             0.7740272846536966,
                             0.8062226189304507,
                             0.11558772111174682,
                             0.6208644434783503,
                             -0.8669876949524729,
                             -0.39303334375412996,
                             0.190505957037845,
                             0.2646568572274895,
                             0.4622508068490933,
                             -0.6125692153936919,
                             0.94868727864955,
                             0.7542289805470132,
                             0.4659695770901731,
                             -0.2698439135252697,
                             -0.7135900559558361,
                             0.9278323910011149,
                             -0.7817909011480624,
                             -0.46703840614192865,
                             -0.06301429695981907,
                             -0.17455647565639487,
                             0.7305966762646748,
                             0.4598482614546683,
                             0.5468558141576126,
                             0.44234365104881385,
                             -0.24910043569715912,
                             0.35578641156843344,
                             -0.9413735018785268,
                             -0.7248934085769465,
                             -0.4790798535341261,
                             -0.6779935362216386,
                             -0.7858406010666572,
                             -0.8097400805782412,
                             0.1906318422758979,
                             0.8999064063878153,
                             -0.2054116690580115,
                             0.6351521263659594,
                             0.13031707236075324,
                             -0.10028943287446267,
                             -0.20242631719999463,
                             -0.6505679810043801,
                             0.4448308181773236)
    val data = Frame(
      Mat(y, x),
      Index(0 until 50: _*),
      Index("y", "x1")
    )

    val fit = LogisticRegression.logisticRegression(data, "y")

    val fit1 =
      sgd.Sgd
        .optimize(data,
                  "y",
                  sgd.LogisticRegression,
                  sgd.L2(0d),
                  sgd.NewtonUpdater,
                  normalize = false)
        .get

    assert(
      fit1.estimatesV.roundTo(5) == Vec(-0.0148874586, -0.2756813856547127)
        .roundTo(5))

    assert(
      fit.covariate("intercept").get._1.slope.roundTo(5) == -0.0148874586
        .roundTo(5))
    assert(
      fit.covariate("intercept").get._1.sd.roundTo(5) == 0.28408100490251437
        .roundTo(5))
    assert(
      fit.covariate("x1").get._1.slope.roundTo(5) == -0.2756813856547127
        .roundTo(5))
    assert(fit.covariate("x1").get._1.sd.roundTo(5) == 0.49353.roundTo(5))
    assert(fit.logLikelihood.L.roundTo(5) == -34.501741139860144.roundTo(5))
  }
}
