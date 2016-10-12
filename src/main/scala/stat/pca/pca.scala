package stat.pca

import org.saddle._

case class PCA[RX: ORD: ST, CX: ORD: ST](eigenvalues: Vec[Double],
                                         projected: Frame[RX, Int, Double],
                                         loadings: Frame[CX, Int, Double])
