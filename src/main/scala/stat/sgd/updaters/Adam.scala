package stat.sgd

import org.saddle._
import org.saddle.linalg._
import stat.matops._

case class AdamStep(point: Vec[Double],
                    acc1: Vec[Double],
                    acc2: Vec[Double],
                    b1Acc: Double,
                    b2Acc: Double)
    extends ItState

case class AdamUpdater(lr: Double = 0.0001) extends Updater[AdamStep] {
  def next[M: MatOps](b: Vec[Double],
                      batch: Batch[M],
                      obj: ObjectiveFunction[_, _],
                      pen: Penalty[_],
                      last: Option[AdamStep]) = {

    val penalizationMask = obj.adaptPenalizationMask(batch)
    val j = obj.jacobi(b, batch) - pen.jacobi(b, penalizationMask)

    val mt = last.map(_.acc1).getOrElse(vec.zeros(j.length))
    val vt = last.map(_.acc2).getOrElse(vec.zeros(j.length))
    val bt1 = last.map(_.b1Acc).getOrElse(1d)
    val bt2 = last.map(_.b2Acc).getOrElse(1d)

    val b1 = 0.9
    val b2 = 0.999
    val eps = 1E-8

    val mtp1 = mt * b1 + j * (1 - b1)
    val vtp1 = vt * b2 + j.map(x => x * x * (1 - b2))

    val btp1 = b1 * bt1
    val btp2 = b2 * bt2

    val mtp1hat = mtp1 / (1 - btp1)
    val vtp1hat = vtp1 / (1 - btp2)

    val update = j * mtp1hat.zipMap(vtp1hat)((m, v) =>
        lr * m / (math.sqrt(v) + eps))

    val next = b + update

    AdamStep(next, mtp1, vtp1, btp1, btp2)

  }
}
