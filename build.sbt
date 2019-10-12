scalaVersion := "2.12.9"

name := "stat"

organization := "io.github.pityka"

version := "0.0.10-SNAPSHOT"

libraryDependencies ++= Seq(
  "io.github.pityka" %% "saddle-linalg" % "2.0.0-M3",
  "io.github.pityka" %% "saddle-stats" % "2.0.0-M3",
  "io.github.pityka" % "hierarchical-clustering-fork" % "1.0-5",
  "io.github.pityka" %% "nspl-saddle" % "0.0.22-SNAPSHOT",
  "io.github.pityka" %% "nspl-awt" % "0.0.22-SNAPSHOT" % "test",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test",
  "biz.enef" %% "slogging" % "0.6.1",
  "com.lihaoyi" %% "upickle" % "0.7.1"
)

parallelExecution in Test := false

publishTo := sonatypePublishTo.value

lazy val root = project in file(".")

lazy val example1 = project
  .in(file("examples/regression"))
  .settings(
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "nspl-awt" % "0.0.20",
      "io.github.pityka" %% "fileutils" % "1.2.2",
      "com.lihaoyi" %% "upickle" % "0.7.1"
    ),
    scalaVersion := "2.12.9"
  )
  .dependsOn(root)

lazy val example2 = project
  .in(file("examples/classification"))
  .settings(
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "nspl-awt" % "0.0.20",
      "io.github.pityka" %% "fileutils" % "1.2.2",
      "com.lihaoyi" %% "upickle" % "0.7.1"
    ),
    scalaVersion := "2.12.9"
  )
  .dependsOn(root)

pomExtra in Global := {
  <url>https://pityka.github.io/stat</url>
  <licenses>
    <license>
      <name>MIT</name>
      <url>https://opensource.org/licenses/MIT</url>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:github.com/pityka/stat</connection>
    <developerConnection>scm:git:git@github.com:pityka/stat</developerConnection>
    <url>github.com/pityka/stat</url>
  </scm>
  <developers>
    <developer>
      <id>pityka</id>
      <name>Istvan Bartha</name>
      <url>https://pityka.github.io/stat/</url>
    </developer>
  </developers>
}
